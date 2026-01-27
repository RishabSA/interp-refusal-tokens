import json
from functools import partial
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformer_lens.utils import get_act_name
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer
from lm_eval.api.model import LM
from lm_eval import evaluator

from scripts.hooked_model import load_hooked_model
from scripts.steering import steering_hook
from scripts.linear_probe import (
    LinearProbe,
    get_categorical_steering_vector_probe,
    get_low_rank_combination_steering_probe,
)
from scripts.low_rank_combination_steering import LowRankSteeringCombination


class TLensSteeredLM(LM):
    def __init__(
        self,
        hooked_model: HookedTransformer,
        tokenizer: PreTrainedTokenizerBase,
        mode: str = "baseline",  # baseline, categorical_steering, low_rank_combination
        layer: int = 18,
        activation_name: str = "resid_post",
        append_seq: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        max_length: int = 4096,
        benign_strength: float = -6.0,
        harmful_strength: float = 4.0,
        steering_vector_mapping: dict[int, torch.Tensor] = None,
        low_rank_combination: LowRankSteeringCombination = None,
        probe_model: LinearProbe = None,
        probe_threshold: float = 0.5,
        probe_X_mean: torch.Tensor = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        self.model = hooked_model
        self.tokenizer = tokenizer
        self.mode = mode
        self.layer = layer
        self.activation_name = activation_name
        self.hook_name = get_act_name(activation_name, layer)
        self.append_seq = append_seq
        self.max_length = max_length
        self.benign_strength = benign_strength
        self.harmful_strength = harmful_strength
        self.steering_vector_mapping = steering_vector_mapping
        self.low_rank_combination = low_rank_combination
        self.probe_model = probe_model
        self.probe_threshold = probe_threshold
        self.probe_X_mean = probe_X_mean
        self.device = device

        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        self._route_cache = {}

    def _trim_until(self, text, until_list):
        if not until_list:
            return text

        if isinstance(until_list, str):
            until_list = [until_list]

        cut = None
        for u in until_list:
            if not u:
                continue

            idx = text.find(u)

            if idx != -1:
                if cut is None or idx < cut:
                    cut = idx

        if cut is None:
            return text

        return text[:cut]

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, v):
        self._max_length = v

    @property
    def max_gen_toks(self):
        # LM Eval Harness will override this if the generation cap is not specified
        return 256

    @property
    def batch_size(self):
        # LM Eval Harness will override this if batch_size is passed in
        return 1

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, d):
        self._device = d

    def _get_hooks_for_context(self, context_str):
        if self.mode == "baseline":
            return []

        if context_str in self._route_cache:
            steering_obj, strength, kind = self._route_cache[context_str]
        else:
            routed_prompt = context_str + self.append_seq

            if self.mode == "categorical_steering":
                if self.steering_vector_mapping is None or self.probe_model is None:
                    raise ValueError(
                        "categorical_steering requires steering_vector_mapping and probe_model"
                    )

                steering_obj, strength = get_categorical_steering_vector_probe(
                    prompt=routed_prompt,
                    hooked_model=self.model,
                    benign_strength=self.benign_strength,
                    harmful_strength=self.harmful_strength,
                    steering_vector_mapping=self.steering_vector_mapping,
                    probe_model=self.probe_model,
                    probe_X_mean=self.probe_X_mean,
                    probe_threshold=self.probe_threshold,
                    activation_name=self.activation_name,
                    layer=self.layer,
                    device=self.device,
                )

                kind = "vector"
            elif self.mode == "low_rank_combination":
                if self.low_rank_combination is None or self.probe_model is None:
                    raise ValueError(
                        "low_rank_combination requires low_rank_combination and probe_model"
                    )

                steering_obj, strength = get_low_rank_combination_steering_probe(
                    prompt=routed_prompt,
                    hooked_model=self.model,
                    benign_strength=self.benign_strength,
                    harmful_strength=self.harmful_strength,
                    low_rank_combination=self.low_rank_combination,
                    probe_model=self.probe_model,
                    probe_X_mean=self.probe_X_mean,
                    probe_threshold=self.probe_threshold,
                    activation_name=self.activation_name,
                    layer=self.layer,
                    device=self.device,
                )

                kind = "lowrank"
            else:
                raise ValueError(f"Unknown mode specified: {self.mode}")

            self._route_cache[context_str] = (steering_obj, strength, kind)

        if kind == "vector":
            # steering_obj can be None if something went wrong
            if steering_obj is None:
                return []

            vec = steering_obj.to(
                self.device, dtype=next(self.model.parameters()).dtype
            )

            hook_fn = partial(steering_hook, vec, None, strength)

            return [(self.hook_name, hook_fn)]

        if kind == "lowrank":
            hook_fn = partial(steering_hook, None, steering_obj, strength)

            return [(self.hook_name, hook_fn)]

        return []

    def _unwrap_ll_req(self, req):
        if hasattr(req, "args"):
            return req.args[0], req.args[1]

        return req[0], req[1]

    def _unwrap_gen_req(self, req):
        if hasattr(req, "args"):
            args = req.args
            kwargs = getattr(req, "kwargs", {}) or {}

            prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
            until = args[1] if len(args) > 1 else kwargs.get("until", None)
            max_gen_toks = (
                args[2] if len(args) > 2 else kwargs.get("max_gen_toks", None)
            )

            return prompt, until, max_gen_toks

        prompt = req.get("prompt")
        until = req.get("until", None)
        max_gen_toks = req.get("max_gen_toks", None)

        return prompt, until, max_gen_toks

    def loglikelihood(self, requests):
        out = []

        for req in tqdm(requests, desc="loglikelihood", leave=False):
            context, continuation = self._unwrap_ll_req(req)

            full = context + continuation
            toks = self.model.to_tokens(full).to(self.device)
            ctx_toks = self.model.to_tokens(context).to(self.device)
            ctx_len = ctx_toks.shape[1]

            hooks = self._get_hooks_for_context(context)

            with torch.inference_mode():
                if hooks:
                    with self.model.hooks(hooks):
                        logits = self.model(toks)
                else:
                    logits = self.model(toks)

            logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = toks[:, 1:]

            start = max(ctx_len - 1, 0)
            cont_targets = targets[:, start:]

            cont_lp = (
                logprobs[:, start:, :]
                .gather(-1, cont_targets.unsqueeze(-1))
                .squeeze(-1)
            )

            ll = cont_lp.sum().item()
            out.append((ll, True))

            self.model.reset_hooks()

        return out

    def loglikelihood_rolling(self, requests):
        out = []

        for req in requests:
            s = req.args[0] if hasattr(req, "args") else req

            # Approximate by scoring whole string as continuation from empty context
            ll, _ = self.loglikelihood([("", s)])[0]
            out.append((ll, True))

        return out

    def generate_until(self, requests):
        res = []

        for req in tqdm(requests, desc="generate_until", leave=False):
            prompt, until, max_gen_toks = self._unwrap_gen_req(req)
            if max_gen_toks is None:
                max_gen_toks = self.max_gen_toks

            toks = self.model.to_tokens(prompt).to(self.device)
            hooks = self._get_hooks_for_context(prompt)

            with torch.inference_mode():
                if hooks:
                    with self.model.hooks(hooks):
                        gen = self.model.generate(
                            toks,
                            max_new_tokens=max_gen_toks,
                            do_sample=False,
                            temperature=0.0,
                            return_type="str",
                            stop_at_eos=True,
                        )
                else:
                    gen = self.model.generate(
                        toks,
                        max_new_tokens=max_gen_toks,
                        do_sample=False,
                        temperature=0.0,
                        return_type="str",
                        stop_at_eos=True,
                    )

            self.model.reset_hooks()

            gen_text = gen
            if gen_text.startswith(prompt):
                gen_text = gen_text[len(prompt) :]

            gen_text = self._trim_until(gen_text, until)
            res.append(gen_text)

        return res


def run_lm_eval_harness_steered(
    mode: str = "baseline",  # baseline, categorical_steering, low_rank_combination
    tasks: str = "mmlu,truthfulqa_mc1,truthfulqa_mc2,hellaswag,arc_challenge,piqa",
    num_fewshot: int | None = None,
    batch_size: int = 1,
    model_id: str = "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens",
    layer: int = 18,
    activation_name: str = "resid_post",
    append_seq: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    benign_strength: float = -6.0,
    harmful_strength: float = 4.0,
    steering_vector_mapping: dict[int, torch.Tensor] = None,
    low_rank_combination: LowRankSteeringCombination = None,
    probe_model: LinearProbe = None,
    probe_threshold: float = 0.5,
    probe_X_mean: torch.Tensor = None,
    output_json_path: str = "lm_eval_harness_steered_outputs.json",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
    hooked_model, tokenizer = load_hooked_model(model_id)

    lm = TLensSteeredLM(
        hooked_model=hooked_model,
        tokenizer=tokenizer,
        mode=mode,
        layer=layer,
        activation_name=activation_name,
        append_seq=append_seq,
        benign_strength=benign_strength,
        harmful_strength=harmful_strength,
        steering_vector_mapping=steering_vector_mapping,
        low_rank_combination=low_rank_combination,
        probe_model=probe_model,
        probe_threshold=probe_threshold,
        probe_X_mean=probe_X_mean,
        device=device,
    )

    tasks = [task.strip() for task in tasks.split(",") if task.strip()]
    print(f"Running tasks: {tasks}")
    print(f"Mode: {mode} | num_fewshot={num_fewshot} | batch_size={batch_size}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        log_samples=True,
        verbosity="INFO",
        # limit=50,
    )

    print("Results:")
    print(json.dumps(results["results"], indent=2))

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print(f"\nSaved full JSON output results to: {output_json_path}")

    return results
