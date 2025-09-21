from __future__ import annotations

from contextlib import nullcontext
from typing import List, Dict, Any

import torch
from torch import amp
import torch.nn.functional as F

from transformer_lens.utils import get_act_name  # requires transformer_lens

# lm-evaluation-harness
from lm_eval.api.model import LM
from lm_eval import evaluator, tasks


def _autocast_ctx(device: torch.device | None):
    if device is not None:
        return amp.autocast(device.type, dtype=torch.float16)
    return nullcontext()


def _build_steering_hook_with_positions(sv_batch: torch.Tensor, strength: float, pos_idx: torch.Tensor):
    """
    sv_batch: [B, D] (zeros for samples with no steering)
    pos_idx:  [B] positions to steer (usually last context token index per sample)
    """

    def hook_fn(activation, hook):
        # activation: [batch_size, seq_len, d_model]
        B, S, D = activation.shape
        out = activation
        sv = sv_batch.to(activation.device, dtype=activation.dtype)  # [B, D]
        idx = torch.arange(B, device=activation.device)
        out[idx, pos_idx, :] = out[idx, pos_idx, :] + float(strength) * sv
        return out

    return hook_fn


class HookedSteeredLM(LM):
    def __init__(
        self,
        hooked_model,
        tokenizer,
        get_steering_vector,
        strength: float = -5.0,
        layer: int = 9,
        act_name: str = "resid_post",
        max_gen_tokens: int = 256,
        device: torch.device | None = None,
        batch_size: int = 8,
    ):
        super().__init__()

        self.hm = hooked_model
        self.tok = tokenizer
        self.get_sv = get_steering_vector
        self.strength = float(strength)
        self.layer = int(layer)
        self.act_name = str(act_name)
        self.max_gen = int(max_gen_tokens)
        self._bs = int(batch_size)
        self.device = device if device is not None else self.hm.cfg.device

        # harness queries these:
        self.EOT_TOKEN_ID = self.tok.eos_token_id
        self._max_length = getattr(self.hm.cfg, "n_ctx", 2048)

    # ---- required LM API ----
    @property
    def eot_token_id(self):
        return self.EOT_TOKEN_ID

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._bs

    def tok_encode(self, s: str):
        return self.tok.encode(s, add_special_tokens=False)

    def tok_decode(self, ids):
        return self.tok.decode(ids)

    # ---- core helpers ----
    def _pick_sv_batch(self, prompts: List[str]) -> torch.Tensor:
        """Return [B, D] steering vectors (zeros where None)."""
        vecs = []
        D = None

        for p in prompts:
            v = self.get_sv(p, self.hm) if self.get_sv is not None else None
            if v is None:
                if D is None:
                    D = getattr(self.hm.cfg, "d_model", None)
                if D is None:
                    raise ValueError("Cannot infer d_model to build zero SV.")
                vecs.append(torch.zeros(D))
            else:
                v = v.detach().float().cpu()
                D = v.numel()
                vecs.append(v)

        return torch.stack(vecs, dim=0)  # [B, D]

    def _add_steering_hooks(self, sv_batch: torch.Tensor, pos_idx: torch.Tensor):
        hname = get_act_name(self.act_name, self.layer)
        hook = _build_steering_hook_with_positions(sv_batch, self.strength, pos_idx)
        return [(hname, hook)]

    # :loglikelihood: used by MMLU/TruthfulQA-MC
    def loglikelihood(self, requests: List[tuple[str, str]]):
        """
        requests: list of (context_str, continuation_str)
        returns: list of (sum_logprob, is_greedy)
        """
        outs = []
        self.hm.eval()

        for i in range(0, len(requests), self._bs):
            chunk = requests[i : i + self._bs]
            contexts = [c for (c, _) in chunk]
            conts = [x for (_, x) in chunk]
            prompts = [c for c in contexts]  # steering decision uses context

            # tokenize separately to compute positions
            ctx_tok = self.tok(contexts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            cont_tok = self.tok(conts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            B = len(chunk)
            ctx_lens = (ctx_tok.attention_mask.sum(dim=1)).tolist()
            cont_lens = (cont_tok.attention_mask.sum(dim=1)).tolist()

            # Build full input = context + continuation (no special tokens)
            full_ids = []
            for j in range(B):
                full = torch.cat([
                    ctx_tok.input_ids[j, : ctx_lens[j]],
                    cont_tok.input_ids[j, : cont_lens[j]],
                ], dim=0)
                full_ids.append(full)

            maxlen = max(x.size(0) for x in full_ids)
            pad_id = self.tok.pad_token_id or self.tok.eos_token_id
            full_batch = torch.full((B, maxlen), pad_id, dtype=torch.long, device=self.device)
            attn_mask = torch.zeros((B, maxlen), dtype=torch.long, device=self.device)
            for j, ids in enumerate(full_ids):
                L = ids.size(0)
                full_batch[j, :L] = ids
                attn_mask[j, :L] = 1

            # Positions to steer = last context token index per sample
            pos_idx = torch.tensor([cl - 1 for cl in ctx_lens], device=self.device, dtype=torch.long)
            sv_batch = self._pick_sv_batch(prompts).to(self.device)

            with torch.inference_mode(), _autocast_ctx(torch.device(self.device)):
                fwd_hooks = self._add_steering_hooks(sv_batch, pos_idx)
                with self.hm.hooks(fwd_hooks):
                    logits = self.hm(full_batch)  # [B, S, V]

            # compute loglikelihood of continuation
            logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)  # Predict next
            sum_lp = []
            is_greedy = []
            for j in range(B):
                start = ctx_lens[j] - 1  # Next token predicted at this index
                end = ctx_lens[j] + cont_lens[j] - 1

                target = full_batch[j, ctx_lens[j] : ctx_lens[j] + cont_lens[j]]
                lp = logprobs[j, start:end, :].gather(1, target.unsqueeze(1)).squeeze(1)
                sum_lp.append(float(lp.sum().item()))
                is_greedy.append(False)

            outs.extend(list(zip(sum_lp, is_greedy)))

            del ctx_tok, cont_tok, full_batch, attn_mask, logits

        return outs

    # Generate_until: used by GSM8k
    def generate_until(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        requests: list of dicts with keys:
        - "context": str
        - "until": list[str]
        - "max_generation_length": int (optional)
        returns: list[str]
        """
        results: List[str] = []
        self.hm.eval()

        for i in range(0, len(requests), self._bs):
            chunk = requests[i : i + self._bs]
            contexts = [r["context"] for r in chunk]
            untils = [r.get("until", []) for r in chunk]
            max_new = [r.get("max_generation_length", self.max_gen) for r in chunk]

            # Per-sample steering vectors (decide from context)
            sv_batch = self._pick_sv_batch(contexts).to(self.device)

            # Tokenize
            tok = self.tok(contexts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            ctx_lens = (tok.attention_mask.sum(dim=1)).tolist()
            pos_idx = torch.tensor([cl - 1 for cl in ctx_lens], device=self.device, dtype=torch.long)

            with torch.inference_mode(), _autocast_ctx(torch.device(self.device)):
                fwd_hooks = self._add_steering_hooks(sv_batch, pos_idx)
                with self.hm.hooks(fwd_hooks):
                    torch.manual_seed(0)
                    gens = self.hm.generate(
                        tok.input_ids,
                        max_new_tokens=max(max_new),
                        do_sample=False,
                        return_type="str",
                        stop_at_eos=True,
                    )

            # Post-process per sample: cut at first stop string if present
            for j, gen_text in enumerate(gens):
                out = gen_text
                for stop in untils[j]:
                    k = out.find(stop)
                    if k != -1:
                        out = out[:k]
                        break
                results.append(out)

            del tok, gens

        return results
