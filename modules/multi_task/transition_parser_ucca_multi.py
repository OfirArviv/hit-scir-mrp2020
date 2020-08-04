from typing import Optional, Dict
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Metric

from modules import StackRnn
from modules.transition_parser_ucca import TransitionParser


@Model.register("transition_parser_ucca_multi_task")
class UccaTransitionParserMultiTaskWrapper(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 word_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 ratio_dim: int,
                 num_layers: int,
                 actions_namespace: str,
                 mces_metric: Metric = None,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 input_dropout: float = 0.0,
                 lemma_text_field_embedder: TextFieldEmbedder = None,
                 pos_tag_embedding: Embedding = None,
                 action_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 text_field_embedder: TextFieldEmbedder = None,
                 stack: Model = None,
                 buffer: Model = None
                 ):
        super().__init__(vocab, regularizer)
        self.args = {
            "vocab": vocab,
            "word_dim": word_dim,
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "ratio_dim": ratio_dim,
            "num_layers": num_layers,
            "mces_metric": mces_metric,
            "recurrent_dropout_probability": recurrent_dropout_probability,
            "layer_dropout_probability": layer_dropout_probability,
            "same_dropout_mask_per_instance": same_dropout_mask_per_instance,
            "input_dropout": input_dropout,
            "lemma_text_field_embedder": lemma_text_field_embedder,
            "pos_tag_embedding": pos_tag_embedding,
            "action_embedding": action_embedding,
            "initializer": initializer,
            "regularizer": regularizer,
            "actions_namespace": actions_namespace,
            "text_field_embedder": text_field_embedder,
            "buffer": buffer,
            "stack": stack,
        }

    def create(self, text_field_embedder: TextFieldEmbedder = None,
                stack: Seq2SeqEncoder = None, buffer: Seq2SeqEncoder = None):
        args = self.args
        if buffer:
            args["buffer"] = buffer

        if stack:
            args["stack"] = stack

        if text_field_embedder:
            args["text_field_embedder"] = text_field_embedder

        return TransitionParser(**args)

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
