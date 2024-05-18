"""General-purpose utility functions"""
from collections import defaultdict
import itertools
from typing import Callable, Dict, Iterable, Sequence, Tuple, Type, TypeVar

import torch

_T = TypeVar("_T")
_U = TypeVar("_U")


def identity(obj: _T,
             ) -> _T:
    """
    The identity function
    :param obj: any object
    :return: the input object
    """
    return obj


def group_by_type(elements: Iterable[_T],
                  key: Callable[[_T], _U] = identity,
                  types: Iterable[Type[_U]] = None,
                  ) -> Dict[Type[_U], Sequence[_T]]:
    """
    Group elements by their type
    :param elements: iterable over elements
    :param key: key function
    :param types: when not null, then only check for these types
    :return: element groupings by type
    """
    def get_type(obj: _U,
                 ) -> Type[_U]:
        if types is None:
            return type(obj)

        for type_category in types:
            if isinstance(obj, type_category):
                return type_category

        return None

    groups = defaultdict(list)
    for element in elements:
        element_type = get_type(key(element))
        if element_type is not None:
            groups[element_type].append(element)
    return dict(groups)


def cosine_similarity(vectors: torch.Tensor,
                      eps: float = 1e-15,
                      ) -> torch.Tensor:
    """
    Compute the cosine similarity of each vector with each other
    :param vectors: input vectors
    :param eps: small value to avoid division by zero
    :return: cosine similarities
    """
    numerator = vectors @ vectors.T
    l2 = torch.mul(vectors, vectors).sum(dim=-1)
    denominator = torch.max(torch.sqrt(torch.outer(l2, l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)


def create_batches(elements: Iterable[_T],
                   batch_size: int,
                   ) -> Iterable[Sequence[_T]]:
    """
    Create batches from iterable
    :param elements: elements
    :param batch_size: batch size
    :return: batches
    """
    try:
        return itertools.batched(elements, batch_size)
    except AttributeError:  # fallback for Python below 3.12
        elem_it = iter(elements)
        while batch := tuple(itertools.islice(elem_it, batch_size)):
            yield batch


def scale_down(current_size: Tuple[int, int],
               target_size: Tuple[int, int],
               ) -> Tuple[int, int]:
    """
    Compute downscaled sizes of an image. If image already smaller than target,
    then the size is unmodified.
    :param current_size: current image size
    :param target_size: target size
    :return: downscaled sizes of an image
    """
    if 0 in target_size:
        return 0, 0

    current_width, current_height = current_size
    target_width, target_height = target_size
    width_factor = current_width / target_width
    height_factor = current_height / target_height
    scale_factor = max(width_factor, height_factor, 1.)
    return (int(current_width  / scale_factor),
            int(current_height / scale_factor))
