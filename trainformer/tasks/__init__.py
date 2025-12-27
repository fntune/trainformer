"""Training tasks."""
from trainformer.tasks.multimodal.clip import CLIP
from trainformer.tasks.multimodal.vlm import VLM
from trainformer.tasks.nlp.causal_lm import CausalLM
from trainformer.tasks.nlp.masked_lm import MaskedLM
from trainformer.tasks.nlp.seq2seq import Seq2Seq
from trainformer.tasks.vision.classification import ImageClassification
from trainformer.tasks.vision.metric_learning import MetricLearning
from trainformer.tasks.vision.ssl import SSL

__all__ = [
    "CLIP",
    "CausalLM",
    "ImageClassification",
    "MaskedLM",
    "MetricLearning",
    "Seq2Seq",
    "SSL",
    "VLM",
]
