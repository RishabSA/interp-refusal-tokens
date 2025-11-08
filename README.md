# What Do Refusal Tokens Learn? Fine-Grained Representations and Evidence for Downstream Steering

**Accepted to the 2025 NeurIPS workshop on Mechanistic Interpretability**

OpenReview: https://openreview.net/forum?id=szBGSWqwB7

Paper PDF: https://openreview.net/pdf?id=szBGSWqwB7

We study whether categorical refusal tokens enable controllable and interpretable safety behavior in language models. Using a fine-tuned version of Llama-3 8B with categorical refusal tokens, we extract residual‑stream activations, compute sparse category‑specific steering vectors, and apply categorical steering at inference time to control refusal behavior. We employ this approach to reduce over-refusal on benign and ambiguous prompts to nearly 0, while maintaining or improving refusal on truly harmful prompts, with no degradation in overall model performance. Model diffing of steering vectors reveals low cross-model cosine similarity for four of the five categories, suggesting that the emergence of our refusal features is mediated by refusal token fine-tuning. Our preliminary results indicate that refusal tokens are promising for shaping fine-grained safety directions that facilitate targeted control and nuanced interpretability, especially for reducing over-refusal while preserving general model capabilities and safety.
