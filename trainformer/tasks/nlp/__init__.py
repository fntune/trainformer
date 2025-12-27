"""NLP tasks."""
from trainformer.tasks.nlp.causal_lm import CausalLM
from trainformer.tasks.nlp.masked_lm import MaskedLM
from trainformer.tasks.nlp.seq2seq import Seq2Seq

__all__ = ["CausalLM", "Seq2Seq", "MaskedLM"]
