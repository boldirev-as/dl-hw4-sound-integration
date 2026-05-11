import torch
from torch import nn
from torch.nn import functional as F

from .modules import QuantizerOutput


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        embedding_dim,
        codebook_size,
        decay,
        eps,
        replace_threshold,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.replace_threshold = replace_threshold

        embedding = torch.randn(codebook_size, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embedding_avg", embedding.clone())
        self.initialized = False

    @torch.no_grad()
    def _kmeans_init(self, flat_x):
        if self.initialized:
            return
        perm = torch.randperm(flat_x.size(0), device=flat_x.device)
        centers = flat_x[perm[: self.codebook_size]].clone()
        for _ in range(10):
            distances = (
                flat_x.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat_x @ centers.t()
                + centers.pow(2).sum(dim=1)
            )
            assignments = distances.argmin(dim=1)
            for index in range(self.codebook_size):
                mask = assignments == index
                if mask.any():
                    centers[index] = flat_x[mask].mean(dim=0)
        self.embedding.copy_(centers)
        self.embedding_avg.copy_(centers)
        self.initialized = True

    def forward(self, x):
        flat_x = x.permute(0, 2, 1).reshape(-1, self.embedding_dim)
        if self.training and not self.initialized and flat_x.size(0) >= self.codebook_size:
            self._kmeans_init(flat_x)

        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_size).type_as(flat_x)
        quantized = F.embedding(encoding_indices, self.embedding).view(x.size(0), x.size(2), x.size(1))
        quantized = quantized.permute(0, 2, 1)

        if self.training:
            assignment_count = encodings.sum(dim=0)
            dw = encodings.t() @ flat_x
            self.cluster_size.mul_(self.decay).add_(assignment_count, alpha=1.0 - self.decay)
            self.embedding_avg.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

            cluster_size = (
                (self.cluster_size + self.eps)
                / (self.cluster_size.sum() + self.codebook_size * self.eps)
                * self.cluster_size.sum()
            )
            normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.copy_(normalized)

            stale_codes = self.cluster_size < self.replace_threshold
            if stale_codes.any():
                random_indices = torch.randint(0, flat_x.size(0), (int(stale_codes.sum().item()),), device=flat_x.device)
                replacement = flat_x[random_indices]
                self.embedding[stale_codes] = replacement
                self.embedding_avg[stale_codes] = replacement
                self.cluster_size[stale_codes] = self.replace_threshold

        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return quantized, perplexity, encoding_indices.view(x.size(0), x.size(2))


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        codebook_size,
        num_quantizers,
        decay,
        eps,
        replace_threshold,
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizerEMA(
                    embedding_dim=embedding_dim,
                    codebook_size=codebook_size,
                    decay=decay,
                    eps=eps,
                    replace_threshold=replace_threshold,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, x):
        residual = x
        quantized_sum = torch.zeros_like(x)
        commitment_loss = x.new_zeros(())
        perplexities = []
        code_indices = []

        for quantizer in self.quantizers:
            quantized, perplexity, codes = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            perplexities.append(perplexity)
            code_indices.append(codes)

        commitment_loss = F.mse_loss(x, quantized_sum.detach())
        codebook_loss = F.mse_loss(quantized_sum, x.detach())
        quantized_st = x + (quantized_sum - x).detach()
        stacked_codes = torch.stack(code_indices, dim=1)
        mean_perplexity = torch.stack(perplexities).mean()
        return QuantizerOutput(
            quantized=quantized_st,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=mean_perplexity,
            codes=stacked_codes,
        )
