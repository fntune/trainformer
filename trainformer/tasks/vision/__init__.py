"""Vision tasks."""
from trainformer.tasks.vision.classification import ImageClassification
from trainformer.tasks.vision.metric_learning import MetricLearning
from trainformer.tasks.vision.ssl import SSL

__all__ = ["ImageClassification", "MetricLearning", "SSL"]
