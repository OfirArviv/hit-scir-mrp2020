import logging
import math

import torch
from typing import Dict
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models.model import Model, remove_pretrained_embedding_params
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.nn.util import get_text_field_mask

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextFieldEmbedderWithEncoder(TextFieldEmbedder):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder
                 ):
        super().__init__()
        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._sub_modules = torch.nn.ModuleList([text_field_embedder, encoder])

    def forward(self,  # pylint: disable=arguments-differ
                text_field_input: Dict[str, torch.Tensor],
                num_wrapping_dims: int = 0,
                **kwargs) -> torch.Tensor:
        embedd = self._text_field_embedder.forward(text_field_input, num_wrapping_dims, **kwargs)

        mask = get_text_field_mask(text_field_input)

        encoded = self._encoder(embedd, mask)

        return encoded

    def get_output_dim(self) -> int:
        return self._encoder.get_output_dim()


class TextFieldEmbedderWithTaskEmbedding(TextFieldEmbedder):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 task_embedding: Embedding,
                 task_index: int,
                 ):
        super().__init__()
        self._text_field_embedder: TextFieldEmbedder = text_field_embedder
        self._task_embedding = task_embedding,
        self._task_index = task_index

        self._sub_modules = torch.nn.ModuleList([text_field_embedder, self._task_embedding[0]])

    def forward(self,  # pylint: disable=arguments-differ
                text_field_input: Dict[str, torch.Tensor],
                num_wrapping_dims: int = 0,
                **kwargs) -> torch.Tensor:
        text_embedding = self._text_field_embedder.forward(text_field_input, num_wrapping_dims, **kwargs)
        task_embedding_input = text_embedding.new_full(text_embedding.shape[:-1], self._task_index).long()
        task_embedding = self._task_embedding[0](task_embedding_input.cuda(0)).to(text_embedding.device)

        output = torch.cat((task_embedding, text_embedding), 2)

        return output

    def get_output_dim(self) -> int:
        return self._text_field_embedder.get_output_dim() + self._task_embedding[0].embedding_dim


# GitHub issue: How to do multi-task learning in AllenNLP? #2618
# https://github.com/allenai/allennlp/issues/2618

# Ensemble model for inspiration:
# https://github.com/allenai/allennlp/blob/f60d826307243f5f5d9ccdb05903dfaa4a5055ca/allennlp/models/ensemble.py


@Model.register("multi-task")
class MultiTaskModel(Model):
    """
    This model does nothing interesting, but it's designed to
    operate on heterogeneous instances using shared parameters
    (well, one shared parameter) like you'd have in multi-task training.
    """

    def __init__(self,
                 models: Dict[str, Model],
                 shared_text_field_embedder: TextFieldEmbedder,
                 shared_encoder: Seq2SeqEncoder = None,
                 task_1_encoder: Seq2SeqEncoder = None,
                 task_2_encoder: Seq2SeqEncoder = None,
                 shared_stack: Model = None,
                 shared_buffer: Model = None,
                 task_embedding_dim: int = None,
                 ) -> None:
        vocab = list(models.values())[0].vocab
        for submodel in models.values():
            if submodel.vocab != vocab:
                raise ConfigurationError("Vocabularies in multil task model differ")

        super().__init__(vocab)

        # Using ModuleList propagates calls to .eval() so dropout is disabled on the submodels in evaluation
        # and prediction.

        text_field_embedder_dict = {task_label: shared_text_field_embedder for task_label in models.keys()}

        if shared_encoder:
            text_field_embedder_dict = {
                task_label: TextFieldEmbedderWithEncoder(text_field_embedder_dict[task_label],
                                                         shared_encoder)
                for task_label in models.keys()}

        if task_embedding_dim:
            task_to_index_dict = vocab.get_token_to_index_vocabulary("task")
            if not all(task_label in task_to_index_dict for task_label in models.keys()):
                print(f'task_to_index_dict:{task_to_index_dict.keys()}')
                print(f'models.keys(): {models.keys()}')

            assert all(task_label in task_to_index_dict for task_label in models.keys())

            weight = torch.FloatTensor(len(models), task_embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=True)
            self.register_parameter(name='embedding_weights', param=self.weight)
            torch.nn.init.xavier_uniform_(self.weight)
            task_embedding = Embedding(num_embeddings=len(models), embedding_dim=task_embedding_dim,
                                       weight=weight)

            text_field_embedder_dict = {
                task_label: TextFieldEmbedderWithTaskEmbedding(text_field_embedder_dict[task_label],
                                                               task_embedding,
                                                               task_to_index_dict[task_label])
                for task_label in models.keys()}

        if task_1_encoder and task_2_encoder:
            task_to_index_dict = vocab.get_token_to_index_vocabulary("task")
            if not all(task_label in task_to_index_dict for task_label in models.keys()):
                print(f'task_to_index_dict:{task_to_index_dict.keys()}')
                print(f'models.keys(): {models.keys()}')

            assert all(task_label in task_to_index_dict for task_label in models.keys())

            assert len(task_to_index_dict) == 2

            task_encoders = [task_1_encoder, task_2_encoder]
            text_field_embedder_dict = {
                task_label: TextFieldEmbedderWithEncoder(text_field_embedder_dict[task_label],
                                                         task_encoder)
                for task_encoder, task_label in zip(task_encoders, models.keys())}

        self._text_field_embedder = torch.nn.ModuleDict(text_field_embedder_dict)

        assert (shared_buffer is None) == (shared_stack is None)
        if shared_stack and shared_buffer:
            shared_stack_buffer = {"stack": shared_stack, "buffer": shared_buffer}
            self._shared_stack_buffer = torch.nn.ModuleDict(shared_stack_buffer)

            models = {
                k: models[k].create(self._text_field_embedder[k],
                                    self._shared_stack_buffer["stack"],
                                    self._shared_stack_buffer["buffer"], )
                for k in models.keys()}
        else:
            models = {k: models[k].create(self._text_field_embedder[k])
                      for k in models.keys()}

        self._models = torch.nn.ModuleDict(models)

    def forward(  # type: ignore
            self,
            **args
    ) -> Dict[str, torch.Tensor]:
        framework = args["framework"][0]
        assert all(instance_framework == framework for instance_framework in args["framework"])
        model = self._models[framework]
        del args["framework"]
        output_dict = model.forward(**args)

        output_dict["framework"] = framework

        if torch.isnan(output_dict["loss"]):
            output_dict["loss"] = torch.zeros_like(output_dict["loss"])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            metric_dict = {f: self._models[f].get_metrics(reset=reset) for f in self._models.keys()}
            for framework, metrics in metric_dict.items():
                for m_key, m_val in metrics.items():
                    all_metrics[f'{framework}_{m_key}'] = m_val

            metric_list = list(metric_dict.values())
            for key in metric_list[0]:
                all_metrics[f'average_{key}'] = sum([m[key] for m in metric_list]) / len(metric_list)

        return all_metrics

    @classmethod
    def _load(cls, config: Params, serialization_dir: str, weights_file: str = None, cuda_device: int = -1) -> "Model":

        # Multi Task Models don't have vocabularies or weights of their own, so they override _load.
        # Taken from :
        # https://github.com/allenai/allennlp/blob/f60d826307243f5f5d9ccdb05903dfaa4a5055ca/allennlp/models/ensemble.py
        model_params = config.get("model")

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        model = Model.from_params(vocab=None, params=model_params)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        return model
