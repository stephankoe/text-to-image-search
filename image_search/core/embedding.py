"""Encode text or images as vector space embedding"""
import abc
from operator import itemgetter
from typing import Generic, Iterable, List, Literal, Sequence, Tuple, TypeVar

from PIL import Image
import torch
from transformers import (AutoTokenizer, CLIPModel, CLIPConfig,
                          CLIPImageProcessor)

from image_search.core.utils import group_by_type

_T = TypeVar("_T")
Distance = Literal["cosine", "dot", "euclid", "manhattan"]


class Embedder(abc.ABC, Generic[_T]):
    """Embed text or images in vector space"""

    distance: Distance
    embedding_dim: int

    @abc.abstractmethod
    def __call__(self,
                 objs: Iterable[_T],
                 ) -> Sequence[torch.Tensor]:
        """
        Get embeddings for texts and/or images
        :param inputs: input objects
        :return: embeddings for input objects
        """
        raise NotImplemented


class CLIPEmbedder(Embedder):
    """Use CLIP compatible model to embed text and/or images in vector space"""

    _INPUT_TYPE = str | Image.Image
    distance = "cosine"

    def __init__(self,
                 model_path: str,
                 device: str | torch.device = torch.device("cpu"),
                 ):
        """
        :param model_path: path to transformers model folder
        :param device: device (e.g., cpu, cuda, ...)
        """
        self._process_image = CLIPImageProcessor.from_pretrained(model_path)
        self._tokenize = AutoTokenizer.from_pretrained(model_path)
        self._device = device
        self._model = CLIPModel.from_pretrained(model_path).to(self._device)
        self._model = self._model.eval()
        self._config = CLIPConfig.from_pretrained(model_path)

    def __call__(self,
                 inputs: Iterable[_INPUT_TYPE] = (),
                 ) -> Sequence[torch.Tensor]:
        """
        Get embeddings for texts and/or images
        :param texts: input texts
        :param images: input images
        :return: embeddings for input
        """
        texts_with_ids, images_with_ids = self._categorize_inputs(inputs)

        embeddings_with_ids = []
        if texts_with_ids:
            text_ids, texts = zip(*texts_with_ids)
            text_inputs = self._tokenize(text=texts,
                                         return_tensors="pt",
                                         padding=True,
                                         )
            text_embedding = self._model.get_text_features(**text_inputs)
            embeddings_with_ids.append((text_embedding, text_ids))

        if images_with_ids:
            image_ids, images = zip(*images_with_ids)
            image_inputs = self._process_image(images=images,
                                               return_tensors="pt",
                                               )
            image_embedding = self._model.get_image_features(**image_inputs)
            embeddings_with_ids.append((image_embedding, image_ids))

        embeddings = self._restore_order(*embeddings_with_ids)
        return embeddings

    @property
    def embedding_dim(self):
        """
        :return: embedding dimension
        """
        return self._config.vision_config.projection_dim

    @staticmethod
    def _categorize_inputs(objs: Iterable[_INPUT_TYPE],
                           ) -> Tuple[Sequence[Tuple[int, str]],
                                      Sequence[Tuple[int, Image.Image]]]:
        """
        Split the list into a list with texts and images
        :param objs: list of input objects
        :return: tuple containing (1) list with texts from input, and
                 (2) list with images from input
        """
        groupings = group_by_type(enumerate(objs),
                                  key=itemgetter(1),
                                  types={str, Image.Image},
                                  )
        return groupings.get(str, ()), groupings.get(Image.Image, ())

    @staticmethod
    def _restore_order(*elements_with_ids: Tuple[Sequence[_T], Sequence[int]],
                       ) -> Sequence[_T]:
        """
        Restore the original order of the elements
        :param elements_with_ids: tuple of elements and IDs
        :return: elements in original order
        """
        n_elements = sum(len(ids) for _, ids in elements_with_ids)
        output: List[_T] = [None] * n_elements
        for elements, ids in elements_with_ids:
            for element, element_id in zip(elements, ids):
                output[element_id] = element
        return output
