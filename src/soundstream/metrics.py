import torch
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


class MetricTracker:
    def __init__(self, sample_rate, use_nisqa):
        self.stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=False)
        self.nisqa = NonIntrusiveSpeechQualityAssessment(sample_rate) if use_nisqa else None
        self.sample_rate = sample_rate

    def update(self, prediction, target):
        prediction = prediction.squeeze(1).detach().cpu()
        target = target.squeeze(1).detach().cpu()
        n = min(prediction.size(-1), target.size(-1))
        prediction = prediction[..., :n]
        target = target[..., :n]
        metrics = {"stoi": float(self.stoi(prediction, target).item())}
        if self.nisqa is not None:
            metrics["nisqa"] = float(self.nisqa(prediction).mean().item())
        return metrics


def average_metrics(items):
    keys = sorted(items[0].keys())
    return {key: sum(item[key] for item in items) / len(items) for key in keys}
