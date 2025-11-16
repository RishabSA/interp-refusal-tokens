import torch
import torch.nn.functional as F


def compute_contrastive_steering_vectors(
    benign_dict: dict[str, torch.Tensor],
    harmful_dict: dict[str, torch.Tensor],
    K: int | None = 100,
    tau: float | None = 1e-3,
) -> dict[str, torch.Tensor]:
    steering_vectors = {}

    # Enforce sparsity by only keeping the top-K values and setting the others to 0
    def get_topk_sparse_vector(vector, K):
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0

        return vector * mask

    # L2 Normalization
    def l2_norm(vector, eps=1e-8):
        return vector / (vector.norm(dim=-1, keepdim=True) + eps)

    for (
        (harmful_category, harmful),
        (benign_category, benign),
    ) in zip(
        harmful_dict.items(),
        benign_dict.items(),
    ):
        if harmful_category != benign_category:
            print("Error: harmful and benign are not the same category")
            break

        steering_harmful = (harmful - benign).mean(dim=0)

        if tau is not None:
            # Filter out inactive features with values < tau
            # boolean mask of shape (d_model)

            tau_mask = steering_harmful.abs() >= tau

            # Convert the bool masks to float masks to multiply
            tau_mask = tau_mask.float()

            # Apply the masks to each of the mean features
            steering_harmful = steering_harmful * tau_mask

        if K is not None:
            steering_harmful = get_topk_sparse_vector(steering_harmful, K)

        steering_harmful = l2_norm(steering_harmful)

        steering_vectors[harmful_category] = steering_harmful

    return steering_vectors


def compute_contrastive_steering_vectors_whitened(
    benign_dict: dict[str, torch.Tensor],
    harmful_dict: dict[str, torch.Tensor],
    K: int | None = 100,
    tau: float | None = 1e-3,
    should_whiten: bool = True,
    shrink_lam: float = 0.1,  # covariance shrinkage toward diagonal
    remove_shared_directions: bool = True,
    should_orthogonalize: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    # Enforce sparsity by only keeping the top-K values and setting the others to 0
    def get_topk_sparse_vector(vector, K):
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0

        return vector * mask

    # L2 Normalization
    def l2_norm(vector, eps=1e-8):
        return vector / (vector.norm(dim=-1, keepdim=True) + eps)

    # Prepare pooled activations for whitening
    # We estimate covariance on pooled (harmful - benign) samples per category
    # (or at least on all benign+harmful activations if shapes mismatch).
    pooled = []
    for cat in harmful_dict.keys():
        H = harmful_dict[cat]  # (N_h, d)
        B = benign_dict[cat]  # (N_b, d)

        pooled.append(H - B)

    X = torch.cat(pooled, dim=0)  # (N_total, d)
    X = X.to(torch.float32)

    # Compute mean and center
    mu = X.mean(0, keepdim=True)  # (1, d)
    Xc = X - mu

    # Covariance with simple shrinkage
    if should_whiten:
        # Empirical covariance
        cov = (Xc.t() @ Xc) / max(Xc.shape[0] - 1, 1)  # (d, d)

        # Shrink towards the diagonal for stability
        diag = torch.diag(torch.diag(cov))
        cov = (1 - shrink_lam) * cov + shrink_lam * diag

        # Jitter
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        # Inverse sqrt via eig
        w, V = torch.linalg.eigh(cov)  # ascending
        W_inv_sqrt = V @ torch.diag(w.clamp_min(1e-12).rsqrt()) @ V.t()

        def whiten(vector):
            return W_inv_sqrt @ vector

    # Raw per-category vectors (mean difference)
    raw_vectors = {}
    for (cat_h, H), (cat_b, B) in zip(harmful_dict.items(), benign_dict.items()):
        if cat_h != cat_b:
            raise ValueError("Harmful and benign keys must align (same categories).")
        # mean difference prototype (robust & simple)
        v = H.mean(0) - B.mean(0)  # (d,)
        raw_vectors[cat_h] = v

    whitened_vectors = {}
    for cat, vector in raw_vectors.items():
        if should_whiten:
            vector = whiten(vector - mu.squeeze(dim=0))
        else:
            vector = vector - mu.squeeze(dim=0)

        if tau is not None:
            # Filter out inactive features with values < tau
            mask = (vector.abs() >= tau).float()
            vector = vector * mask

        if K is not None:
            # Top-K sparsity filtering
            vector = get_topk_sparse_vector(vector, K)

        whitened_vectors[cat] = l2_norm(vector)

    # Remove the shared refusal directions between steering vectors
    decorelated_vectors = dict(whitened_vectors)
    if remove_shared_directions:
        # Get the L2 normalized average of all of the category-specific steering vectors
        generic_vector = l2_norm(
            torch.stack(list(whitened_vectors.values()), dim=0).mean(dim=0)
        )

        for cat, vector in whitened_vectors.items():
            decorelated_vector = (
                vector - (vector @ generic_vector) * generic_vector
            )  # drop the shared directions
            decorelated_vectors[cat] = l2_norm(decorelated_vector)

    # Optional Gram–Schmidt orthogonalization
    final_vectors = dict(decorelated_vectors)
    if should_orthogonalize:
        categories = list(decorelated_vectors.keys())
        basis = []

        for category in categories:
            vector = decorelated_vectors[category].clone()

            for b in basis:
                vector = vector - (vector @ b) * b

            vector = l2_norm(vector)
            basis.append(vector)

            final_vectors[category] = vector

    return final_vectors


def compute_old_steering_vectors(
    mean_benign_dict: dict[str, torch.Tensor],
    mean_harmful_dict: dict[str, torch.Tensor],
    should_filter_shared: bool = False,
    K: int | None = None,
    tau: float | None = None,
) -> dict[str, torch.Tensor]:
    steering_vectors = {}

    # Enforce sparsity by only keeping the top-K values and setting the others to 0
    def get_topk_sparse_vector(vector, K):
        vals, idxs = torch.topk(vector.abs(), K)
        mask = torch.zeros_like(vector)
        mask[idxs] = 1.0

        return vector * mask

    # Normalize the steering vectors to have magnitude = 1
    def normalize_steering_vector(vector):
        norm = vector.norm()

        # Prevent division by 0 error
        return vector / norm if norm > 0 else vector

    for (
        (harmful_category, mean_harmful),
        (benign_category, mean_benign),
    ) in zip(
        mean_harmful_dict.items(),
        mean_benign_dict.items(),
    ):
        if harmful_category != benign_category:
            print("Error: harmful and benign are not the same category")
            break

        if tau is not None:
            # Filter out inactive features with values < tau
            # boolean mask of shape (d_model)

            benign_mask = mean_benign.abs() >= tau
            harmful_mask = mean_harmful.abs() >= tau

        if should_filter_shared:
            # Filter out features that are shared between the mean category-specific harmful activations and the benign activations to isolate behavior-specific components
            harmful_mask = harmful_mask & (~benign_mask)

        if tau is not None or should_filter_shared:
            # Convert the bool masks to float masks to multiply
            benign_mask = benign_mask.float()
            harmful_mask = harmful_mask.float()

            # Apply the masks to each of the mean features
            mean_benign = mean_benign * benign_mask
            mean_harmful = mean_harmful * harmful_mask

        # Subtract the mean benign activations from the mean category-specific harmful activations to get the steering vector for the specific category

        steering_harmful = mean_harmful - mean_benign

        if K is not None:
            steering_harmful = get_topk_sparse_vector(steering_harmful, K)

        steering_harmful = normalize_steering_vector(steering_harmful)

        steering_harmful_cosine_sim = F.cosine_similarity(
            mean_harmful, mean_benign, dim=-1, eps=1e-8
        )
        print(
            f"Harmful category {harmful_category} has cosine similarity of {steering_harmful_cosine_sim} with benign"
        )

        steering_vectors[harmful_category] = steering_harmful

    return steering_vectors
