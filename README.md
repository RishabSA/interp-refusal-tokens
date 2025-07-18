# Interpreting, Steering, and Patching Refusal in LLMs with Categorical Refusal Tokens

Algoverse AI Research Project

Start with the Llama-3 8B model fine-tuned with [refuse_…] tokens for each separate harmful refusal category and [respond] tokens for benign prompts
The fine-tuned models are taken from:
“REFUSAL TOKENS: A SIMPLE WAY TO CALIBRATE REFUSALS IN LARGE LANGUAGE MODELS” - https://arxiv.org/pdf/2412.06748
Here, the authors show that individual category refusal tokens are separable
However, the authors note that the humanizing category may fall into multiple categories and be entangled
Datasets
CoCoNot is used for the harmful and ambiguous sets
Split the harmful prompts into different categories based on the specific refusal types
The specific refusal type is what should be produced as the refusal token by the fine-tuned model
Each set of harmful prompts given by category is Hc
Should generate category-specific refusal tokens
The ambiguous set is given as A
Ambiguous prompts that cause the model to produce either a refuse or respond token due to ambiguous wording
Hold out a fraction of the CoCoNot dataset for later evaluation
Ultrachat, MMLU, GSM8k, and TruthfulQA are used for the benign set
The benign set is given as B
Benign prompts that should generate the response token
It is also used to ensure that the model's performance is not heavily affected by steering refusal features
OR-Bench will also be used as a benchmark for over-refusal
We will evaluate on OR-Bench to test over-refusal rates on the base models and the steered models
It will be used as a held-out validation set for hyperparameter tuning
We can also synthesize our data or a custom benchmark for ambiguous prompts using an LLM
We can also generate ambiguous prompts by taking a harmful prompt and using an LLM to rewrite it just enough to flip it to benign and vice versa
Ensures that ambiguous prompts are truly on the decision boundary
Use few-shot prompting with previously human-edited examples
The goal is to have 50% of these ambiguous prompts belong to each class
To begin with, the LLM should have around a 0% refusal accuracy on these prompts, generating refusal and response tokens for the wrong prompts
Set up a loop to ensure that after every prompt is edited, the refusal/response token flips
If it does not, feed back into the LLM
Each of the dataset evaluations and benchmarks will be used at multiple stages
We will evaluate the pre-trained LLM, the fine-tuned LLM, and the LLM with an attached SAE for steering or patching
Build and train a top-K Sparse Autoencoder (SAE) at layer l of the residual stream with a high expansion factor
We train an SAE to make sure that the specific refusal features and the benign features are as decorrelated as possible
When we apply activation steering and attribution patching, we want to make sure that features are decorrelated
Minimize the effect on other behaviors (we evaluate this by testing on general, benign benchmarking datasets)
The SAE only takes in the activation values for the last token in the sequence provided
Represents the next token to be generated
(batch_size, seq_len, d_model) -> (batch_size, d_model)
Layer l has to be chosen
Llama-3 8B has 32 total transformer blocks
Experiment with different layers, such as layers 4, 6, 8, or 10
The authors of the CAA paper state that they “find that behavioral clustering emerges around one-third of the way through the layers for the behaviors that they study”
Layers 10-15 in Llama 2 Chat
Use a high expansion factor in the SAE
“Steering Language Model Refusal with Sparse Autoencoders” - https://arxiv.org/pdf/2411.11296
The authors use an expansion factor of 8
They find that the expansion factor might have been too low
Choose a high enough expansion factor that ensures that features are not correlated with each other
It is necessary when we try to separate features and group them into features for specific refusal types
Experiment with different values for the SAE expansion factor, such as 8, 12, or 16
The SAE loss function has the following parts:
Trained to minimize the reconstruction loss (MSE) between the input and output activations
||x-x||2
Trained with a L1 sparsity constraint
||z||1
Where:
controls the sparsity strength
The final loss function is defined as: L=||x-x||2+||z||1
If the separation of features fails, it's more likely that either the model is unable to separate them, or the data is not accentuating the separation we want to see
We could also experiment with Supervised Contrastive Loss or a multi-term loss function that pushes harmful and benign activations apart
We will identify and evaluate category-specific refusal features on the residual stream activations of the LLM and the sparse activations of an SAE trained on the residual stream activations of the LLM
We test to see if using an SAE makes the category-specific refusal features better separated and less entangled
For both attribution patching and activation steering, with and without an SAE, we can identify refusal features in 2 ways:
Identify steering vectors with SAS + CAA
“Steering Llama 2 via Contrastive Activation Addition” - https://arxiv.org/pdf/2312.06681
“Steering Large Language Model Activations in Sparse Spaces” - https://arxiv.org/pdf/2503.00177
Combine the methods of Contrastive Activation Addition (CAA) and Sparse Activation Steering (SAS) to identify and steer the features with and without an SAE
The authors of the CAA paper state that they “find that behavioral clustering emerges around one-third of the way through the layers for the behaviors that they study”
Refusal is one of the behaviors that they study
They also observe “linear separability of residual stream activations in two dimensions emerging suddenly after a particular layer)
For their testing, with refusal behavior on Llama 2 7B Chat, they found that refusal behavior became linearly separable at layer 10
At layer 9, features were still very correlated and entangled

Chart from the CAA paper
The process of computing steering vectors for each refusal category Hc is given as:
Given 2 sets of prompts:
1 set of prompts Hc are the refusal prompts for the specific category c
The other set of prompts B are the benign prompts
Pass both sets of prompts through the LLM up till layer l
Cache the activations a of the last token in each sequence at layer l
When identifying features with an SAE, pass the activations for each set of prompts through the SAE encoder
Cache the sparse activations z of the last token in each sequence at layer l
For residual-stream activations (no SAE)
Compute the mean category-specific harmful activations as:
aHc=1|aHc|aHci aHcaHci
Compute the mean benign activations as:
aB=1|aB|aBi aBaBi
For sparse activations (SAE)
Compute the mean category-specific harmful sparse activations as:
zHc=1|zHc|zHci zHczHci
Compute the mean benign sparse activations as:
zB=1|zB|zBi zBzBi
Filter out inactive features with a threshold
Filter out all features with values <
Remove features that are shared between the mean category-specific harmful activations and the benign activations to isolate behavior-specific components
Subtract the mean benign activations from the mean category-specific harmful activations to get the steering vector for the specific category c
For residual-stream activations (no SAE)
va(c,l)=aHc-aB
For sparse activations (SAE)
vz(c,l)=zHc-zB
These steering vectors have the identified features for the specific category c as non-zero values
We could also choose the top-K features and set all other features to 0 to get a strict set of features
Trained Classification Heads
The classification head can be done with a logistic regression or a separately trained classification neural network
Take each of the SAE sparse vectors from the benign prompts B and the harmful prompts for each of the C refusal categories H1 through HC
Make an input features vector XRnM where each row is an SAE sparse vector for all of the benign and harmful prompts
Where M is the size of the sparse latent vectors
n is the number of total benign and harmful prompts
Make a target class vector yRn where each item is the class of the corresponding input
yi=0 when the class is a benign prompt
yi={1,,C} when the class is a harmful prompt in the class c
Train a logistic regression with softmax for multinomial using Scikit-Learn, or train a neural network with softmax
Use an L1 regularization penalty for sparsity on weights
Fits a weight matrix WRc+1M
For each class (benign or harmful for category c), take the coefficient row wjRM
For class j=0 (benign class), pick the top-K indices with the largest positive weights in w0
Use these as the benign features
For class j=c (harmful class for category c), pick the top-K indices with the largest positive weights in jc
Use these as the refusal features for class c
Experiment with attribution patching
Perform attribution patching with and without an SAE to identify specific refusal features
“Attribution Patching Outperforms Automated Circuit Discovery” - https://arxiv.org/pdf/2310.10348  
Attribution patching is a gradient-based approach that approximates activation patching
Activation patching requires 1 forward pass on the clean prompt, 1 forward pass on the corrupted prompt, and 1 forward pass for every patched prompt
Attribution patching estimates the causal effect of patching with 1 forward pass on the clean prompt, 1 forward pass on the corrupted prompt, and 1 backward pass
Given a clean prompt
Exhibits the specific refusal category behavior
Record the model activations eclean
Given a corrupted prompt
A benign/harmless prompt
Record the model activations ecorrupt
Approximate the loss shift
LecleanL(xclean)(ecorr-eclean)
Compute eL(xclean) by backpropagating from the loss L back to each activation eclean
Compute the difference vector e=ecorr-eclean
Take the dot product
Outputs the attribution score as a single scalar that estimates the amount that the activations would have changed the loss if they had been directly patched
attribution score(e) =(ecorr-eclean)ecleanL
L(xclean|do(E=ecorr))L(xclean)+(ecorr-eclean)ecleanL(xclean|do(E=eclean))
Where the attribution score is given as:
(ecorr-eclean)ecleanL(xclean| do(E=eclean))
Experiment with activation steering
Perform activation steering with and without an SAE to identify refusal features
We also test, evaluate, and analyze model performance on refusal and over-refusal benchmarks
“Steering Llama 2 via Contrastive Activation Addition” - https://arxiv.org/pdf/2312.06681
“Steering Large Language Model Activations in Sparse Spaces” - https://arxiv.org/pdf/2503.00177
Combine the methods of Contrastive Activation Addition (CAA) and Sparse Activation Steering (SAS) to identify and steer the features with and without an SAE
Steering refusal features in the residual stream activations
We have already identified the category-specific residual features
Using the identified steering vectors va(c, l) perform activation steering
alt=alt+va(c,l)
Where:
is the steering strength
A positive value of increases the category-specific refusal behavior
A negative value of decreases the category-specific refusal behavior
alt represents the t-th token activations at layer l
Steering refusal features in the sparse SAE activations
We have already identified the category-specific residual features
Using the identified steering vectors vz(c, l) perform activation steering
zlt=zlt+vz(c, l)
Where:
is the steering strength
A positive value of increases the category-specific refusal behavior
A negative value of decreases the category-specific refusal behavior
zlt represents the t-th token sparse activations at layer l in the sparse SAE activations
We can also use the identified steering vectors for each category in the residual stream activation space and the sparse SAE latent space to perform ablation
Set the values of the identified features to 0 in the respective activation space or latent space
Evaluate and analyze the effects of ablation to see if the identified steering vector features are the ideal features
Used to see the effects of disabling refusal in the LLM
Measure how much the refusal rate falls
A high drop means that it is a necessary refusal feature
Perform conditional steering in 2 ways
Use a prompt safety/refusal classifier
Use the Minstral LLM that is fine-tuned on the WildGuard benchmark
The WildGuard LLM outputs a simple binary classification of refuse or don’t refuse
This does not apply to every individual category-specific refusal token
This is okay because the different refusal types are providing us with more fine-grained control to find good steering vectors
When using the WildGuard LLM, we just operate on a global level of refusal steering vectors
We can also use the Llama 3 LLM, which is fine-tuned with the category-specific refusal and response tokens
Pass the given prompt through the LLM, but limit the LLM to only generating a single token
This token will always be either a refusal or a response token
Depending on the specific category of the outputted token, perform conditional steering using the category-specific steering vector
Another approach for conditional steering that doesn’t require a different classifier model
Residual-stream activation space (No SAE)
When passing a prompt through the LLM, the activation is passed through the SAE, and we get an activation vector a
Compute dB=||z-aB||2 and dHc=||z-aHc||2 for each cC
Sparse activation latent space (SAE)
When passing a prompt through the LLM, the activation is passed through the SAE, and we get a sparse latent vector z
Compute dB=||z-zB||2 and dHc=||z-zHc||2 for each cC
For each cC, calculate sc=dBdB+dHc
If sc>Thigh, apply steering to the activations for the specific refusal category c
Represents a harmful prompt
Set the steering strength to a positive number
If sc<Tlow, apply negative steering to the activations for the specific refusal category c
Represents a benign prompt
Set the steering strength to a negative number
Else if Tlow<sc<Thigh, apply partial steering to the activations for the specific refusal category c
Represents an ambiguous prompt
Set the steering strength to a tuned value that minimizes over-refusal
Thigh and Tlow are tuned thresholds that can be grid searched over a held-out validation set
Get the fixed threshold values by finding the one that minimizes the over-refusal rates on OR-Bench

To evaluate the quality of identified steering vectors for each category in the residual-stream activation space and the sparse SAE latent space:
Project each of the top-K refusal features for each category and the top-K benign features into 2D using t-SNE, UMap, and PCA
Visualize the projected refusal features for each category and the benign features
Color-code each feature by category and look for visual cluster separation
Find the centroid (mean) of each cluster (category) and use it to find metrics for the tightness and separation of clusters
Silhouette Score, Davies-Bouldin (DB) Index, and Calinski-Harabasz Index
Each of these metrics is used as a way to quantify and evaluate the quality of steering vectors
Compute these metrics on the original dimension features, not the 2D features, to get the most accurate results
Re-train the SAE multiple times with different random seeds set
Only choose the SAE features that appear to be most impactful across all SAE instances to avoid randomness from weight initialization
The results are averaged across seeds to get an unbiased result
When testing steering and patching the SAE activations, we will evaluate the refusal rates on the harmful, benign, and ambiguous sets
We will also assess over-refusal rates on the OR-Bench benchmark
The benign dataset is used to ensure that the model's performance is not heavily affected by steering refusal features
A risk is that some sparse features may entangle refusal signals and other unrelated signals when steering
Could cause unintended side effects by affecting performance on other tasks
We will test this by measuring changes in general task performance on datasets from the benign set
MMLU, GSM8k, TruthfulQA, etc
Get the Llama-3.1 8B and Mistral-v0.3 models fine-tuned with [refuse_…] tokens for each separate harmful refusal category and [respond] tokens for benign prompts
The fine-tuned models are taken from the “REFUSAL TOKENS: A SIMPLE WAY TO CALIBRATE REFUSALS IN LARGE LANGUAGE MODELS” paper
Use both models to test the generalizability of the steering vectors, SAE, and SAE steering vectors SAE
Directly transfer the steering vectors, SAE, and SAE steering vectors the SAE that are trained on the llama-3 8B LLM to the other 2 LLMs
The hidden sizes (d_model) for all three models are 4096
Feed the activations at layer l of the other 2 LLMs into the SAE
Test to see if the identified refusal features per category and the identified benign features are generalizable
Do latent space visualization, clamping, ablation, and patching to see if the features are consistent across LLMs
Test the generalizability of the residual-stream activations and the sparse SAE latent activations
