# Results

For every model, we evaluated refusal rates on: **`COCONot Original, COCONot Contrast, WildGuard Unharmful, WildGuard Harmful, WildJailbreak Adversarial Benign, WildJailbreak Adversarial Harmful, OR-Bench Hard, and OR-Bench Toxic**` benchmarks.

For models that were fine-tuned to produce refusal tokens, we evaluate refusal rates by checking for the rate of a refusal token being present and with an LLM as a judge.
For models that do not produce refusal tokens, we evaluate refusal rates with an LLM as a judge.

For the LLM as a judge, we use **`GPT-5-mini**`.

## Llama Instruct Baseline

### COCONot Original

#### Refusal Rate by LLM as a Judge

**`367/1001 -> 36.6633%`**
**`374/1001 -> 37.3626%`**
**`361/1001 -> 36.0639%`**

**`36.6966% ± 0.5307%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 262/395 | 66.3291%       | 268/395 | 67.8481%       | 265/395 | 67.0886%       |
| Unsupported requests          | 50/157  | 31.8471%       | 60/157  | 38.2166%       | 58/157  | 36.9427%       |
| Indeterminate requests        | 37/142  | 26.0563%       | 34/142  | 23.9437%       | 30/142  | 21.1268%       |
| Incomplete requests           | 6/225   | 2.6667%        | 8/225   | 3.5556%        | 6/225   | 2.6667%        |
| Humanizing requests           | 12/82   | 14.6341%       | 4/82    | 4.8780%        | 2/82    | 2.4390%        |

### COCONot Contrast

**`11/379 -> 2.9024%`**
**`11/379 -> 2.9024%`**
**`11/379 -> 2.9024%`**

**`2.9024% ± 0.0000%`**

#### Refusal Rate by LLM as a Judge

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 10/149  | 6.7114%        | 10/149  | 6.7114%        | 10/149  | 6.7114%        |
| Unsupported requests          | 0/82    | 0.0000%        | 0/82    | 0.0000%        | 0/82    | 0.0000%        |
| Incomplete requests           | 1/148   | 0.6757%        | 1/148   | 0.6757%        | 1/148   | 0.6757%        |

### WildGuard Unharmful

#### Refusal Rate by LLM as a Judge

**`105/945 -> 11.1111%`**
**`106/945 -> 11.2169%`**
**`105/945 -> 11.1111%`**

**`11.1464% ± 0.0499%`**

### WildGuard Harmful

#### Refusal Rate by LLM as a Judge

**`597/754 -> 79.1777%`**
**`590/754 -> 78.2493%`**
**`599/754 -> 79.4430%`**

**`78.9567% ± 0.5118%`**

### WildJailbreak Adversarial Benign

#### Refusal Rate by LLM as a Judge

**`53/210 -> 25.2381%`**
**`53/210 -> 25.2381%`**
**`53/210 -> 25.2381%`**

**`25.2381% ± 0.0000%`**

### WildJailbreak Adversarial Harmful

#### Refusal Rate by LLM as a Judge

### OR-Bench Hard

#### Refusal Rate by LLM as a Judge

**`820/1319 -> 62.1683%`**
**`833/1319 -> 63.1539%`**
**`841/1319 -> 63.7604%`**

**`63.0275% ± 0.6561%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 365/527 | 69.2600%       | 365/527 | 69.2600%       | 366/527 | 69.4497%       |
| unethical  | 67/125  | 53.6000%       | 69/125  | 55.2000%       | 68/125  | 54.4000%       |
| self-harm  | 43/63   | 68.2540%       | 43/63   | 68.2540%       | 49/63   | 77.7778%       |
| privacy    | 124/199 | 62.3116%       | 121/199 | 60.8040%       | 125/199 | 62.8141%       |
| harassment | 23/41   | 56.0976%       | 23/41   | 56.0976%       | 23/41   | 56.0976%       |
| violence   | 48/66   | 72.7273%       | 48/66   | 72.7273%       | 48/66   | 72.7273%       |
| deception  | 38/72   | 52.7778%       | 39/72   | 54.1667%       | 37/72   | 51.3889%       |
| harmful    | 42/106  | 39.6226%       | 55/106  | 51.8868%       | 55/106  | 51.8868%       |
| hate       | 39/54   | 72.2222%       | 39/54   | 72.2222%       | 39/54   | 72.2222%       |
| sexual     | 31/66   | 46.9697%       | 31/66   | 46.9697%       | 31/66   | 46.9697%       |

### OR-Bench Toxic

#### Refusal Rate by LLM as a Judge

**`605/655 -> 92.3664%`**
**`602/655 -> 91.9084%`**
**`599/655 -> 91.4504%`**

**`91.9084% ± 0.3740%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 47/50   | 94.0000%       | 47/50   | 94.0000%       | 47/50   | 94.0000%       |
| unethical  | 47/61   | 77.0492%       | 44/61   | 72.1311%       | 47/61   | 77.0492%       |
| self-harm  | 92/92   | 100.0000%      | 92/92   | 100.0000%      | 92/92   | 100.0000%      |
| privacy    | 56/61   | 91.8033%       | 56/61   | 91.8033%       | 56/61   | 91.8033%       |
| harassment | 69/76   | 90.7895%       | 69/76   | 90.7895%       | 64/76   | 84.2105%       |
| violence   | 69/71   | 97.1831%       | 69/71   | 97.1831%       | 69/71   | 97.1831%       |
| deception  | 81/83   | 97.5904%       | 81/83   | 97.5904%       | 81/83   | 97.5904%       |
| harmful    | 28/30   | 93.3333%       | 29/30   | 96.6667%       | 28/30   | 93.3333%       |
| hate       | 57/58   | 98.2759%       | 57/58   | 98.2759%       | 57/58   | 98.2759%       |
| sexual     | 59/73   | 80.8219%       | 58/73   | 79.4521%       | 58/73   | 79.4521%       |

## Deepseek Llama Distill Baseline

### COCONot Original

#### Refusal Rate by LLM as a Judge

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns |         |                |         |                |         |                |
| Unsupported requests          |         |                |         |                |         |                |
| Indeterminate requests        |         |                |         |                |         |                |
| Incomplete requests           |         |                |         |                |         |                |
| Humanizing requests           |         |                |         |                |         |                |

### COCONot Contrast

#### Refusal Rate by LLM as a Judge

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns |         |                |         |                |         |                |
| Unsupported requests          |         |                |         |                |         |                |
| Incomplete requests           |         |                |         |                |         |                |

### WildGuard Unharmful

#### Refusal Rate by LLM as a Judge

### WildGuard Harmful

#### Refusal Rate by LLM as a Judge

### WildJailbreak Adversarial Benign

#### Refusal Rate by LLM as a Judge

### WildJailbreak Adversarial Harmful

#### Refusal Rate by LLM as a Judge

### OR-Bench Hard

#### Refusal Rate by LLM as a Judge

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    |         |                |         |                |         |                |
| unethical  |         |                |         |                |         |                |
| self-harm  |         |                |         |                |         |                |
| privacy    |         |                |         |                |         |                |
| harassment |         |                |         |                |         |                |
| violence   |         |                |         |                |         |                |
| deception  |         |                |         |                |         |                |
| harmful    |         |                |         |                |         |                |
| hate       |         |                |         |                |         |                |
| sexual     |         |                |         |                |         |                |

### OR-Bench Toxic

#### Refusal Rate by LLM as a Judge

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    |         |                |         |                |         |                |
| unethical  |         |                |         |                |         |                |
| self-harm  |         |                |         |                |         |                |
| privacy    |         |                |         |                |         |                |
| harassment |         |                |         |                |         |                |
| violence   |         |                |         |                |         |                |
| deception  |         |                |         |                |         |                |
| harmful    |         |                |         |                |         |                |
| hate       |         |                |         |                |         |                |
| sexual     |         |                |         |                |         |                |

## Binary Refusal Baseline

### COCONot Original

#### Refusal Rate by Refusal Token

**`952/1001 -> 95.1049%`**

| Category                      | Ratio   | Refusal Rate |
| ----------------------------- | ------- | ------------ |
| Requests with safety concerns | 389/395 | 98.4810%     |
| Unsupported requests          | 151/157 | 96.1783%     |
| Indeterminate requests        | 124/142 | 87.3239%     |
| Incomplete requests           | 210/225 | 93.3333%     |
| Humanizing requests           | 78/82   | 95.1220%     |

#### Refusal Rate by LLM as a Judge

**`880/1001 -> 87.9121%`**
**`869/1001 -> 86.8132%`**
**`866/1001 -> 86.5135%`**

**`87.0796% ± 0.6012%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 381/395 | 96.4557%       | 386/395 | 97.7215%       | 378/395 | 95.6962%       |
| Unsupported requests          | 138/157 | 87.8981%       | 149/157 | 94.9045%       | 148/157 | 94.2675%       |
| Indeterminate requests        | 120/142 | 84.5070%       | 120/142 | 84.5070%       | 119/142 | 83.8028%       |
| Incomplete requests           | 168/225 | 74.6667%       | 150/225 | 66.6667%       | 160/225 | 71.1111%       |
| Humanizing requests           | 73/82   | 89.0244%       | 64/82   | 78.0488%       | 61/82   | 74.3902%       |

### COCONot Contrast

#### Refusal Rate by Refusal Token

**`72/379 -> 18.9974%`**

| Category                      | Ratio  | Refusal Rate |
| ----------------------------- | ------ | ------------ |
| Requests with safety concerns | 15/149 | 10.0671%     |
| Unsupported requests          | 11/82  | 13.4146%     |
| Incomplete requests           | 46/148 | 31.0811%     |

#### Refusal Rate by LLM as a Judge

**`69/379 -> 18.2058%`**
**`49/379 -> 12.9288%`**
**`64/379 -> 16.8865%`**

**`16.0070% ± 2.2423%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 16/149  | 10.7383%       | 17/149  | 11.4094%       | 16/149  | 10.7383%       |
| Unsupported requests          | 11/82   | 13.4146%       | 9/82    | 10.9756%       | 9/82    | 10.9756%       |
| Incomplete requests           | 42/148  | 28.3784%       | 23/148  | 15.5405%       | 39/148  | 26.3514%       |

### WildGuard Unharmful

#### Refusal Rate by Refusal Token

**`146/945 -> 15.4497%`**

#### Refusal Rate by LLM as a Judge

**`142/945 -> 15.0265%`**
**`136/945 -> 14.3915%`**
**`144/945 -> 15.2381%`**

**`14.8854% ± 0.3597%`**

### WildGuard Harmful

#### Refusal Rate by Refusal Token

**`449/754 -> 59.5491%`**

#### Refusal Rate by LLM as a Judge

**`452/754 -> 59.9469%`**
**`454/754 -> 60.2122%`**
**`453/754 -> 60.0796%`**

**`60.0796% ± 0.1083%`**

### WildJailbreak Adversarial Benign

#### Refusal Rate by Refusal Token

**`11/210 -> 5.2381%`**

#### Refusal Rate by LLM as a Judge

**`12/210 -> 5.7143%`**
**`15/210 -> 7.1429%`**
**`12/210 -> 5.7143%`**

**`6.1905% ± 0.6734%`**

### WildJailbreak Adversarial Harmful

#### Refusal Rate by Refusal Token

**`517/2000 -> 25.8500%`**

#### Refusal Rate by LLM as a Judge

### OR-Bench Hard

#### Refusal Rate by Refusal Token

**`157/1319 -> 11.9030%`**

| Category   | Ratio  | Refusal Rate |
| ---------- | ------ | ------------ |
| illegal    | 49/527 | 9.2979%      |
| unethical  | 16/125 | 12.8000%     |
| self-harm  | 9/63   | 14.2857%     |
| privacy    | 25/199 | 12.5628%     |
| harassment | 9/41   | 21.9512%     |
| violence   | 6/66   | 9.0909%      |
| deception  | 9/72   | 12.5000%     |
| harmful    | 11/106 | 10.3774%     |
| hate       | 11/54  | 20.3704%     |
| sexual     | 12/66  | 18.1818%     |

#### Refusal Rate by LLM as a Judge

**`184/1319 -> 13.9500%`**
**`197/1319 -> 14.9356%`**
**`200/1319 -> 15.1630%`**

**`14.6828% ± 0.5265%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 57/527  | 10.8159%       | 65/527  | 12.3340%       | 65/527  | 12.3340%       |
| unethical  | 18/125  | 14.4000%       | 20/125  | 16.0000%       | 22/125  | 17.6000%       |
| self-harm  | 13/63   | 20.6349%       | 13/63   | 20.6349%       | 15/63   | 23.8095%       |
| privacy    | 33/199  | 16.5829%       | 36/199  | 18.0905%       | 34/199  | 17.0854%       |
| harassment | 10/41   | 24.3902%       | 10/41   | 24.3902%       | 10/41   | 24.3902%       |
| violence   | 7/66    | 10.6061%       | 7/66    | 10.6061%       | 7/66    | 10.6061%       |
| deception  | 9/72    | 12.5000%       | 9/72    | 12.5000%       | 9/72    | 12.5000%       |
| harmful    | 14/106  | 13.2075%       | 11/106  | 10.3774%       | 14/106  | 13.2075%       |
| hate       | 11/54   | 20.3704%       | 11/54   | 20.3704%       | 12/54   | 22.2222%       |
| sexual     | 12/66   | 18.1818%       | 15/66   | 22.7273%       | 12/66   | 18.1818%       |

### OR-Bench Toxic

#### Refusal Rate by Refusal Token

**`467/655 -> 71.2977%`**

| Category   | Ratio | Refusal Rate |
| ---------- | ----- | ------------ |
| illegal    | 28/50 | 56.0000%     |
| unethical  | 30/61 | 49.1803%     |
| self-harm  | 77/92 | 83.6957%     |
| privacy    | 35/61 | 57.3770%     |
| harassment | 47/76 | 61.8421%     |
| violence   | 58/71 | 81.6901%     |
| deception  | 58/83 | 69.8795%     |
| harmful    | 18/30 | 60.0000%     |
| hate       | 50/58 | 86.2069%     |
| sexual     | 66/73 | 90.4110%     |

#### Refusal Rate by LLM as a Judge

**`477/655 -> 72.8244%`**
**`478/655 -> 72.9771%`**
**`474/655 -> 72.3664%`**

**`72.7226% ± 0.2595%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 30/50   | 60.0000%       | 30/50   | 60.0000%       | 28/50   | 56.0000%       |
| unethical  | 31/61   | 50.8197%       | 31/61   | 50.8197%       | 31/61   | 50.8197%       |
| self-harm  | 80/92   | 86.9565%       | 79/92   | 85.8696%       | 79/92   | 85.8696%       |
| privacy    | 35/61   | 57.3770%       | 37/61   | 60.6557%       | 35/61   | 57.3770%       |
| harassment | 47/76   | 61.8421%       | 49/76   | 64.4737%       | 47/76   | 61.8421%       |
| violence   | 59/71   | 83.0986%       | 59/71   | 83.0986%       | 60/71   | 84.5070%       |
| deception  | 59/83   | 71.0843%       | 58/83   | 69.8795%       | 59/83   | 71.0843%       |
| harmful    | 18/30   | 60.0000%       | 18/30   | 60.0000%       | 18/30   | 60.0000%       |
| hate       | 51/58   | 87.9310%       | 52/58   | 89.6552%       | 50/58   | 86.2069%       |
| sexual     | 67/73   | 91.7808%       | 65/73   | 89.0411%       | 67/73   | 91.7808%       |

## Categorical Refusal Baseline

### COCONot Original

#### Refusal Rate by Refusal Token

**`941/1001 -> 94.0060%`**

| Category                      | Ratio   | Refusal Rate |
| ----------------------------- | ------- | ------------ |
| Requests with safety concerns | 386/395 | 97.7215%     |
| Unsupported requests          | 146/157 | 92.9936%     |
| Indeterminate requests        | 132/142 | 92.9577%     |
| Incomplete requests           | 199/225 | 88.4444%     |
| Humanizing requests           | 78/82   | 95.1220%     |

#### Refusal Rate by LLM as a Judge

**`794/1001 -> 79.3207%`**
**`760/1001 -> 75.9241%`**
**`752/1001 -> 75.1249%`**

**`76.7899% ± 1.8190%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 374/395 | 94.6835%       | 373/395 | 94.4304%       | 370/395 | 93.6709%       |
| Unsupported requests          | 146/157 | 92.9936%       | 132/157 | 84.0764%       | 140/157 | 89.1720%       |
| Indeterminate requests        | 87/142  | 61.2676%       | 86/142  | 60.5634%       | 80/142  | 56.3380%       |
| Incomplete requests           | 129/225 | 57.3333%       | 111/225 | 49.3333%       | 126/225 | 56.0000%       |
| Humanizing requests           | 58/82   | 70.7317%       | 58/82   | 70.7317%       | 36/82   | 43.9024%       |

### COCONot Contrast

#### Refusal Rate by Refusal Token

**`43/379 -> 11.3456%`**

| Category                      | Ratio  | Refusal Rate |
| ----------------------------- | ------ | ------------ |
| Requests with safety concerns | 12/149 | 8.0537%      |
| Unsupported requests          | 24/148 | 16.2162%     |
| Incomplete requests           | 7/82   | 8.5366%      |

#### Refusal Rate by LLM as a Judge

**`33/379 -> 8.7071%`**
**`37/379 -> 9.7625%`**
**`30/379 -> 7.9156%`**

**`8.7951% ± 0.7566%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 18/149  | 12.0805%       | 18/149  | 12.0805%       | 16/149  | 10.7383%       |
| Unsupported requests          | 4/82    | 4.8780%        | 7/82    | 8.5366%        | 4/82    | 4.8780%        |
| Incomplete requests           | 11/148  | 7.4324%        | 12/148  | 8.1081%        | 10/148  | 6.7568%        |

### WildGuard Unharmful

#### Refusal Rate by Refusal Token

**`90/945 -> 9.5238%`**

#### Refusal Rate by LLM as a Judge

**`96/945 -> 10.1587%`**
**`101/945 -> 10.6878%`**
**`94/945 -> 9.9471%`**

**`10.2646% ± 0.3115%`**

### WildGuard Harmful

#### Refusal Rate by Refusal Token

**`450/754 -> 59.6817%`**

#### Refusal Rate by LLM as a Judge

**`441/754 -> 58.4881%`**
**`461/754 -> 61.1406%`**
**`435/754 -> 57.6923%`**

**`59.1070% ± 1.4742%`**

### WildJailbreak Adversarial Benign

#### Refusal Rate by Refusal Token

**`7/210 -> 3.3333%`**

#### Refusal Rate by LLM as a Judge

**`14/210 -> 6.6667%`**
**`13/210 -> 6.1905%`**
**`15/210 -> 7.1429%`**

**`6.6667% ± 0.3888%`**

### WildJailbreak Adversarial Harmful

#### Refusal Rate by Refusal Token

**`509/2000 -> 25.4500%`**

#### Refusal Rate by LLM as a Judge

### OR-Bench Hard

#### Refusal Rate by Refusal Token

**`278/1319 -> 21.0766%`**

| Category   | Ratio   | Refusal Rate |
| ---------- | ------- | ------------ |
| illegal    | 105/527 | 19.9241%     |
| unethical  | 25/125  | 20.0000%     |
| self-harm  | 12/63   | 19.0476%     |
| privacy    | 46/199  | 23.1156%     |
| harassment | 11/41   | 26.8293%     |
| violence   | 16/66   | 24.2424%     |
| deception  | 11/72   | 15.2778%     |
| harmful    | 19/106  | 17.9245%     |
| hate       | 22/54   | 40.7407%     |
| sexual     | 11/66   | 16.6667%     |

#### Refusal Rate by LLM as a Judge

**`348/1319 -> 26.3836%`**
**`357/1319 -> 27.0660%`**
**`359/1319 -> 27.2176%`**

**`26.8891% ± 0.3627%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 138/527 | 26.1860%       | 144/527 | 27.3245%       | 140/527 | 26.5655%       |
| unethical  | 33/125  | 26.4000%       | 35/125  | 28.0000%       | 35/125  | 28.0000%       |
| self-harm  | 15/63   | 23.8095%       | 15/63   | 23.8095%       | 15/63   | 23.8095%       |
| privacy    | 58/199  | 29.1457%       | 56/199  | 28.1407%       | 52/199  | 26.1307%       |
| harassment | 11/41   | 26.8293%       | 11/41   | 26.8293%       | 11/41   | 26.8293%       |
| violence   | 21/66   | 31.8182%       | 20/66   | 30.3030%       | 20/66   | 30.3030%       |
| deception  | 15/72   | 20.8333%       | 14/72   | 19.4444%       | 16/72   | 22.2222%       |
| harmful    | 24/106  | 22.6415%       | 26/106  | 24.5283%       | 27/106  | 25.4717%       |
| hate       | 23/54   | 42.5926%       | 23/54   | 42.5926%       | 22/54   | 40.7407%       |
| sexual     | 10/66   | 15.1515%       | 13/66   | 19.6970%       | 21/66   | 31.8182%       |

### OR-Bench Toxic

#### Refusal Rate by Refusal Token

**`533/655 -> 81.3740%`**

| Category   | Ratio | Refusal Rate |
| ---------- | ----- | ------------ |
| illegal    | 38/50 | 76.0000%     |
| unethical  | 45/61 | 73.7705%     |
| self-harm  | 89/92 | 96.7391%     |
| privacy    | 43/61 | 70.4918%     |
| harassment | 61/76 | 80.2632%     |
| violence   | 59/71 | 83.0986%     |
| deception  | 60/83 | 72.2892%     |
| harmful    | 19/30 | 63.3333%     |
| hate       | 55/58 | 94.8276%     |
| sexual     | 64/73 | 87.6712%     |

#### Refusal Rate by LLM as a Judge

**`545/655 -> 83.2061%`**
**`547/655 -> 83.5115%`**
**`527/655 -> 80.4580%`**

**`82.3919% ± 1.3731%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 43/50   | 86.0000%       | 43/50   | 86.0000%       | 43/50   | 86.0000%       |
| unethical  | 42/61   | 68.8525%       | 43/61   | 70.4918%       | 42/61   | 68.8525%       |
| self-harm  | 88/92   | 95.6522%       | 86/92   | 93.4783%       | 88/92   | 95.6522%       |
| privacy    | 46/61   | 75.4098%       | 45/61   | 73.7705%       | 46/61   | 75.4098%       |
| harassment | 61/76   | 80.2632%       | 65/76   | 85.5263%       | 41/76   | 53.9474%       |
| violence   | 63/71   | 88.7324%       | 61/71   | 85.9155%       | 63/71   | 88.7324%       |
| deception  | 67/83   | 80.7229%       | 67/83   | 80.7229%       | 67/83   | 80.7229%       |
| harmful    | 20/30   | 66.6667%       | 20/30   | 66.6667%       | 20/30   | 66.6667%       |
| hate       | 55/58   | 94.8276%       | 55/58   | 94.8276%       | 55/58   | 94.8276%       |
| sexual     | 60/73   | 82.1918%       | 62/73   | 84.9315%       | 62/73   | 84.9315%       |

## Categorical Refusal Steered

### COCONot Original

#### Refusal Rate by Refusal Token

**`943/1001 -> 94.2058%`**

| Category                      | Ratio   | Refusal Rate |
| ----------------------------- | ------- | ------------ |
| Requests with safety concerns | 387/395 | 97.9747%     |
| Unsupported requests          | 148/157 | 94.2675%     |
| Indeterminate requests        | 132/142 | 92.9577%     |
| Incomplete requests           | 198/225 | 88.0000%     |
| Humanizing requests           | 78/82   | 95.1220%     |

#### Refusal Rate by LLM as a Judge

**`762/1001 -> 76.1239%`**
**`799/1001 -> 79.8202%`**
**`772/1001 -> 77.1229%`**

**`77.6890% ± 1.5612%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 376/395 | 95.1899%       | 365/395 | 92.4051%       | 377/395 | 95.4430%       |
| Unsupported requests          | 146/157 | 92.9936%       | 144/157 | 91.7197%       | 130/157 | 82.8025%       |
| Indeterminate requests        | 93/142  | 65.4930%       | 90/142  | 63.3803%       | 97/142  | 68.3099%       |
| Incomplete requests           | 125/225 | 55.5556%       | 129/225 | 57.3333%       | 124/225 | 55.1111%       |
| Humanizing requests           | 22/82   | 26.8293%       | 71/82   | 86.5854%       | 44/82   | 53.6585%       |

### COCONot Contrast

#### Refusal Rate by Refusal Token

**`13/379 -> 3.4301%`**

| Category                      | Ratio  | Refusal Rate |
| ----------------------------- | ------ | ------------ |
| Requests with safety concerns | 1/149  | 0.6711%      |
| Unsupported requests          | 0/82   | 0.0000%      |
| Incomplete requests           | 12/148 | 8.1081%      |

#### Refusal Rate by LLM as a Judge

**`9/379 -> 2.3747%`**
**`6/379 -> 1.5831%`**
**`9/379 -> 2.3747%`**

**`2.1108% ± 0.3731%`**

| Category                      | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ----------------------------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| Requests with safety concerns | 4/149   | 2.6846%        | 1/149   | 0.6711%        | 4/149   | 2.6846%        |
| Unsupported requests          | 0/82    | 0.0000%        | 0/82    | 0.0000%        | 0/82    | 0.0000%        |
| Incomplete requests           | 5/148   | 3.3784%        | 5/148   | 3.3784%        | 5/148   | 3.3784%        |

### WildGuard Unharmful

#### Refusal Rate by Refusal Token

**`17/945 -> 1.7989%`**

#### Refusal Rate by LLM as a Judge

**`38/945 -> 4.0212%`**
**`42/945 -> 4.4444%`**
**`42/945 -> 4.4444%`**

**`4.3034% ± 0.1995%`**

### WildGuard Harmful

#### Refusal Rate by Refusal Token

**`497/754 -> 65.9151%`**

#### Refusal Rate by LLM as a Judge

**`479/754 -> 63.5279%`**
**`481/754 -> 63.7931%`**
**`476/754 -> 63.1300%`**

**`63.4836% ± 0.2725%`**

### WildJailbreak Adversarial Benign

#### Refusal Rate by Refusal Token

**`0/210 -> 0.0000%`**

#### Refusal Rate by LLM as a Judge

**`5/210 -> 2.3810%`**
**`8/210 -> 3.8095%`**
**`5/210 -> 2.3810%`**

**`2.8571% ± 0.6734%`**

### WildJailbreak Adversarial Harmful

#### Refusal Rate by Refusal Token

**`695/2000 -> 34.7500%`**

#### Refusal Rate by LLM as a Judge

### OR-Bench Hard

#### Refusal Rate by Refusal Token

**`264/1319 -> 20.0152%`**

| Category   | Ratio  | Refusal Rate |
| ---------- | ------ | ------------ |
| illegal    | 98/527 | 18.5958%     |
| unethical  | 24/125 | 19.2000%     |
| self-harm  | 11/63  | 17.4603%     |
| privacy    | 43/199 | 21.6080%     |
| harassment | 11/41  | 26.8293%     |
| violence   | 15/66  | 22.7273%     |
| deception  | 11/72  | 15.2778%     |
| harmful    | 19/106 | 17.9245%     |
| hate       | 21/54  | 38.8889%     |
| sexual     | 11/66  | 16.6667%     |

#### Refusal Rate by LLM as a Judge

**`328/1319 -> 24.8673%`**
**`327/1319 -> 24.7915%`**
**`339/1319 -> 25.7013%`**

**`25.1200% ± 0.4122%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 126/527 | 23.9089%       | 121/527 | 22.9602%       | 129/527 | 24.4782%       |
| unethical  | 35/125  | 28.0000%       | 35/125  | 28.0000%       | 34/125  | 27.2000%       |
| self-harm  | 13/63   | 20.6349%       | 12/63   | 19.0476%       | 14/63   | 22.2222%       |
| privacy    | 53/199  | 26.6332%       | 54/199  | 27.1357%       | 62/199  | 31.1558%       |
| harassment | 12/41   | 29.2683%       | 12/41   | 29.2683%       | 11/41   | 26.8293%       |
| violence   | 19/66   | 28.7879%       | 18/66   | 27.2727%       | 20/66   | 30.3030%       |
| deception  | 16/72   | 22.2222%       | 17/72   | 23.6111%       | 14/72   | 19.4444%       |
| harmful    | 17/106  | 16.0377%       | 17/106  | 16.0377%       | 17/106  | 16.0377%       |
| hate       | 24/54   | 44.4444%       | 24/54   | 44.4444%       | 22/54   | 40.7407%       |
| sexual     | 13/66   | 19.6970%       | 17/66   | 25.7576%       | 16/66   | 24.2424%       |

### OR-Bench Toxic

#### Refusal Rate by Refusal Token

**`595/655 -> 90.8397%`**

| Category   | Ratio | Refusal Rate |
| ---------- | ----- | ------------ |
| illegal    | 44/50 | 88.0000%     |
| unethical  | 51/61 | 83.6066%     |
| self-harm  | 91/92 | 98.9130%     |
| privacy    | 52/61 | 85.2459%     |
| harassment | 72/76 | 94.7368%     |
| violence   | 65/71 | 91.5493%     |
| deception  | 73/83 | 87.9518%     |
| harmful    | 23/30 | 76.6667%     |
| hate       | 57/58 | 98.2759%     |
| sexual     | 67/73 | 91.7808%     |

#### Refusal Rate by LLM as a Judge

**`573/655 -> 87.4809%`**
**`578/655 -> 88.2443%`**
**`541/655 -> 82.5954%`**

**`86.1069% ± 2.5025%`**

| Category   | Ratio 1 | Refusal Rate 1 | Ratio 2 | Refusal Rate 2 | Ratio 3 | Refusal Rate 3 |
| ---------- | ------- | -------------- | ------- | -------------- | ------- | -------------- |
| illegal    | 45/50   | 90.0000%       | 43/50   | 86.0000%       | 45/50   | 90.0000%       |
| unethical  | 46/61   | 75.4098%       | 46/61   | 75.4098%       | 9/61    | 14.7541%       |
| self-harm  | 91/92   | 98.9130%       | 89/92   | 96.7391%       | 91/92   | 98.9130%       |
| privacy    | 53/61   | 86.8852%       | 52/61   | 85.2459%       | 52/61   | 85.2459%       |
| harassment | 62/76   | 81.5789%       | 62/76   | 81.5789%       | 60/76   | 78.9474%       |
| violence   | 67/71   | 94.3662%       | 65/71   | 91.5493%       | 66/71   | 92.9577%       |
| deception  | 62/83   | 74.6988%       | 73/83   | 87.9518%       | 74/83   | 89.1566%       |
| harmful    | 23/30   | 76.6667%       | 24/30   | 80.0000%       | 20/30   | 66.6667%       |
| hate       | 58/58   | 100.0000%      | 58/58   | 100.0000%      | 58/58   | 100.0000%      |
| sexual     | 66/73   | 90.4110%       | 66/73   | 90.4110%       | 66/73   | 90.4110%       |
