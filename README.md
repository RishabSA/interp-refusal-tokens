# From Refusal Tokens to Refusal Control: Discovering and Steering Category-Specific Refusal Directions

<img src="resources/interp_refusal_diagram.png" alt="Categorical steering methodology diagram" width="49%"/> <img src="resources/refusal_overrefusal_tradeoff.png" alt="Refusal vs over-refusal tradeoff on models and our methodologies" width="49%"/>

Language models are commonly fine-tuned for safety alignment to refuse harmful prompts. One approach fine-tunes them to generate categorical refusal tokens that distinguish different refusal types before producing a final response. In this work, we leverage a version of Llama 3 8B fine-tuned with these categorical refusal tokens to enable inference-time control over fine-grained refusal behavior, improving both safety and reliability. We show that refusal token fine-tuning induces separable, category-aligned directions in the residual stream, which we extract and use to construct categorical steering vectors with a lightweight probe that determines whether to steer toward or away from refusal during inference. In addition, we introduce a learned low-rank combination that mixes these category directions in a whitened, orthonormal steering basis, resulting in a single controllable intervention under activation-space anisotropy, and show that this intervention is transferable across same-architecture model variants without additional training. Across benchmarks, both categorical steering vectors and the low-rank combination consistently reduce over-refusals on benign prompts while increasing refusal rates on harmful prompts, highlighting their utility for multi-category refusal control.

![Results on safety benchmarks, measuring refusal rates and over-refusal rates using baseline models and our methodologies](resources/safety_results.png)

![Results on general model capability benchmarks using baseline models and our methodologies](resources/general_capability_results.png)

An early version of the paper was **🎉 accepted to the 2025 NeurIPS workshop on Mechanistic Interpretability 🎉** **[OpenReview](https://openreview.net/forum?id=szBGSWqwB7)**, **[Paper PDF](https://openreview.net/pdf?id=szBGSWqwB7)**
