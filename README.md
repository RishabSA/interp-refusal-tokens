# What Do Refusal Tokens Learn? Fine-Grained Representations and Evidence for Downstream Steering

**🎉 Accepted to the 2025 NeurIPS workshop on Mechanistic Interpretability 🎉**

**[OpenReview](https://openreview.net/forum?id=szBGSWqwB7)**

**[Paper PDF](https://openreview.net/pdf?id=szBGSWqwB7)**

![Results of steering over-refusal on an example prompt](resources/interp_refusal_prompt.png)

![Methodology Diagram](resources/interp_refusal_diagram.png)

**Abstract:** Language models are fine-tuned for safety alignment to refuse harmful prompts. One such method involves fine-tuning a language model to generate categorical refusal tokens that distinguish the different types of refusals. In this work, we investigate whether categorical refusal tokens enable controllable, interpretable refusal behavior in language models. Specifically, using a fine-tuned version of Llama-3 8B Base with categorical refusal tokens, we extract residual‑stream activations and compute category‑specific steering vectors. We then apply the category-specific steering vectors at inference-time to control refusal behavior, reducing over-refusals on benign and ambiguous prompts to nearly 0, while maintaining refusal rates on truly harmful prompts and minimizing degradation to general model performance. We perform model diffing of steering vectors between Llama-3 8B Base and the refusal-token fine-tuned model, revealing low cross-model cosine similarity in four of the five categories, suggesting that the emergence of our identified refusal features is mediated specifically by refusal-token fine-tuning. Our results indicate that refusal tokens are promising for shaping fine-grained safety directions that facilitate targeted control, interpretability, and reduced over-refusals.

## Usage

To run all evaluations, steering vector computations, model diffing, steered generations, and more, run from the [Jupyter notebook](refusal_tokens.ipynb), which imports scripts from the scripts folder.

Please send any inquiries to 27ralagharu@woodward.edu

## Citing this work

If you find this work useful in your research, please consider citing our [paper](https://openreview.net/forum?id=szBGSWqwB7):

```tex
@inproceedings{
alagharu2025what,
title={What Do Refusal Tokens Learn? Fine-Grained Representations and Evidence for Downstream Steering},
author={Rishab Alagharu and Ishneet Sukhvinder Singh and Anjali Batta and Jaelyn S. Liang and Shaibi Shamsudeen and Arnav Sheth and Kevin Zhu and Ashwinee Panda and Zhen Wu},
booktitle={Mechanistic Interpretability Workshop at NeurIPS 2025},
year={2025},
url={https://openreview.net/forum?id=szBGSWqwB7}
}
```
