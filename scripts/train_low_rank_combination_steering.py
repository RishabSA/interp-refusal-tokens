import os
from functools import partial
from tqdm.auto import tqdm
from tqdm.contrib import tzip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from scripts.low_rank_combination_steering import LowRankSteeringMap
from scripts.steering import steering_hook


def train_low_rank_combination_steering_map(
    hooked_model: HookedTransformer,
    low_rank_map: LowRankSteeringMap,
    harmful_training_prompts_dataloader: DataLoader,
    benign_training_prompts_dataloader: DataLoader,
    harmful_testing_prompts_dataloader: DataLoader,
    benign_testing_prompts_dataloader: DataLoader,
    prompt_seq_append: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    activation_name: str = "resid_post",
    layer: int = 18,
    strength: float = 1.0,
    kl_loss_weight: float = 1.0,
    epochs: int = 5,
    lr: float = 1e-3,
    eps: float = 1e-12,
    checkpoint_path: str = "low_rank_map_epoch_5.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> LowRankSteeringMap:
    def kl_divergence(
        base_probs: torch.Tensor, steered_probs: torch.Tensor
    ) -> torch.Tensor:
        # base_probs, steered_probs shape: (batch_size, vocab_size)

        return (base_probs * (base_probs / steered_probs).log()).sum(dim=-1).mean()

    def kl_divergence_from_logits(
        base_logits: torch.Tensor,
        steered_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Compute with 32-bit precision for stability
        base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
        steered_log_probs = F.log_softmax(steered_logits.float(), dim=-1)

        base_probs = base_log_probs.exp()

        return (base_probs * (base_log_probs - steered_log_probs)).sum(dim=-1).mean()

    refusal_token_ids = torch.tensor(
        [128256, 128257, 128258, 128259, 128260], device=device, dtype=torch.long
    )

    hooked_model.eval()
    for param in hooked_model.parameters():
        param.requires_grad = False

    hook_name = get_act_name(activation_name, layer)

    optimizer = torch.optim.AdamW(params=low_rank_map.parameters(), lr=lr)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        low_rank_map.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    low_rank_map.to(device).train()

    for epoch in tqdm(range(start_epoch, epochs), desc=f"Training for {epochs} epochs"):
        total_training_harmful_loss = 0.0
        total_training_kl_loss = 0.0
        total_training_loss = 0.0
        num_training_batches = 0

        total_testing_harmful_loss = 0.0
        total_testing_kl_loss = 0.0
        total_testing_loss = 0.0
        num_testing_batches = 0

        training_pbar = tqdm(
            zip(
                harmful_training_prompts_dataloader, benign_training_prompts_dataloader
            ),
            total=min(
                len(harmful_training_prompts_dataloader),
                len(benign_training_prompts_dataloader),
            ),
            desc=f"Training Low-Rank Mapping epoch {epoch + 1}",
        )

        # Training
        for harmful_batch, benign_batch in training_pbar:
            num_training_batches += 1

            harmful_prompts = [
                prompt + prompt_seq_append for prompt in harmful_batch["prompt"]
            ]
            benign_prompts = [
                prompt + prompt_seq_append for prompt in benign_batch["prompt"]
            ]

            harmful_tokens = hooked_model.to_tokens(harmful_prompts).to(device)
            benign_tokens = hooked_model.to_tokens(benign_prompts).to(device)

            hooked_model.reset_hooks()

            # Base
            with torch.no_grad():
                base_benign_logits = hooked_model(benign_tokens)[:, -1, :]
                base_benign_probs = F.softmax(base_benign_logits, dim=-1).clamp_min(
                    eps
                )  # shape: (batch_size, vocab_size)

            # Steered
            hook_fn = partial(
                steering_hook,
                None,
                low_rank_map,
                strength,
            )

            hooked_model.add_hook(hook_name, hook_fn, "fwd")

            steered_benign_logits = hooked_model(benign_tokens)[:, -1, :]
            steered_benign_probs = F.softmax(steered_benign_logits, dim=-1).clamp_min(
                eps
            )  # shape: (batch_size, vocab_size)

            steered_harmful_logits = hooked_model(harmful_tokens)[:, -1, :]
            steered_harmful_probs = F.softmax(steered_harmful_logits, dim=-1).clamp_min(
                eps
            )  # shape: (batch_size, vocab_size)

            # Harmful refusal loss (maximize probability of refusal tokens)
            summed_refusal_probs = steered_harmful_probs.index_select(
                dim=1, index=refusal_token_ids
            ).sum(
                dim=1
            )  # shape: (batch_size)

            harmful_loss = -(summed_refusal_probs.log()).mean()
            total_training_harmful_loss += harmful_loss.item()

            # Benign KL divergence (keep the steered benign distribution close to base benign distribution)
            # kl_loss = kl_divergence(
            #     base_probs=base_benign_probs, steered_probs=steered_benign_probs
            # )

            kl_loss = kl_divergence_from_logits(
                base_logits=base_benign_logits, steered_logits=steered_benign_logits
            )
            total_training_kl_loss += kl_loss.item()

            loss = harmful_loss + kl_loss_weight * kl_loss
            total_training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hooked_model.reset_hooks()

            training_pbar.set_postfix(
                harmful_loss=f"{harmful_loss.item():.6f}",
                kl_loss=f"{kl_loss.item():.6f}",
                loss=f"{loss.item():.6f}",
            )

            del (
                harmful_tokens,
                benign_tokens,
                base_benign_logits,
                base_benign_probs,
                steered_benign_logits,
                steered_benign_probs,
                steered_harmful_logits,
                steered_harmful_probs,
                summed_refusal_probs,
                harmful_loss,
                kl_loss,
                loss,
            )

        testing_pbar = tqdm(
            zip(harmful_testing_prompts_dataloader, benign_testing_prompts_dataloader),
            total=min(
                len(harmful_testing_prompts_dataloader),
                len(benign_testing_prompts_dataloader),
            ),
            desc=f"Testing Low-Rank Mapping epoch {epoch + 1}",
        )

        # Testing
        for harmful_batch, benign_batch in testing_pbar:
            num_testing_batches += 1

            harmful_prompts = [
                prompt + prompt_seq_append for prompt in harmful_batch["prompt"]
            ]
            benign_prompts = [
                prompt + prompt_seq_append for prompt in benign_batch["prompt"]
            ]

            harmful_tokens = hooked_model.to_tokens(harmful_prompts).to(device)
            benign_tokens = hooked_model.to_tokens(benign_prompts).to(device)

            hooked_model.reset_hooks()

            # Base
            with torch.inference_mode():
                base_benign_logits = hooked_model(benign_tokens)[:, -1, :]
                base_benign_probs = F.softmax(base_benign_logits, dim=-1).clamp_min(
                    eps
                )  # shape: (batch_size, vocab_size)

            # Steered
            hook_fn = partial(
                steering_hook,
                None,
                low_rank_map,
                strength,
            )

            hooked_model.add_hook(hook_name, hook_fn, "fwd")

            with torch.inference_mode():
                steered_benign_logits = hooked_model(benign_tokens)[:, -1, :]
                steered_benign_probs = F.softmax(
                    steered_benign_logits, dim=-1
                ).clamp_min(
                    eps
                )  # shape: (batch_size, vocab_size)

                steered_harmful_logits = hooked_model(harmful_tokens)[:, -1, :]
                steered_harmful_probs = F.softmax(
                    steered_harmful_logits, dim=-1
                ).clamp_min(
                    eps
                )  # shape: (batch_size, vocab_size)

            # Harmful refusal loss (maximize probability of refusal tokens)
            summed_refusal_probs = steered_harmful_probs.index_select(
                dim=1, index=refusal_token_ids
            ).sum(
                dim=1
            )  # shape: (batch_size)

            harmful_loss = -(summed_refusal_probs.log()).mean()
            total_testing_harmful_loss += harmful_loss.item()

            # Benign KL divergence (keep the steered benign distribution close to base benign distribution)
            # kl_loss = kl_divergence(
            #     base_probs=base_benign_probs, steered_probs=steered_benign_probs
            # )

            kl_loss = kl_divergence_from_logits(
                base_logits=base_benign_logits, steered_logits=steered_benign_logits
            )
            total_testing_kl_loss += kl_loss.item()

            loss = harmful_loss + kl_loss_weight * kl_loss
            total_testing_loss += loss.item()

            hooked_model.reset_hooks()

            testing_pbar.set_postfix(
                harmful_loss=f"{harmful_loss.item():.6f}",
                kl_loss=f"{kl_loss.item():.6f}",
                loss=f"{loss.item():.6f}",
            )

            del (
                harmful_tokens,
                benign_tokens,
                base_benign_logits,
                base_benign_probs,
                steered_benign_logits,
                steered_benign_probs,
                steered_harmful_logits,
                steered_harmful_probs,
                summed_refusal_probs,
                harmful_loss,
                kl_loss,
                loss,
            )

        print(
            f"Epoch: {epoch + 1} "
            f"| Harmful Training Loss: {total_training_harmful_loss / num_training_batches:.4f} "
            f"| KL Training Loss: {total_training_kl_loss / num_training_batches:.4f} "
            f"| Total Training Loss: {total_training_loss / num_training_batches:.4f}"
            f"| Harmful Testing Loss: {total_testing_harmful_loss / num_testing_batches:.4f} "
            f"| KL Testing Loss: {total_testing_kl_loss / num_testing_batches:.4f} "
            f"| Total Testing Loss: {total_testing_loss / num_testing_batches:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": low_rank_map.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            f"low_rank_map_epoch_{epoch + 1}.pt",
        )

    return low_rank_map
