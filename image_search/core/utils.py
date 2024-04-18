"""General-purpose utility functions"""
from collections import defaultdict
from typing import Callable, Dict, Iterable, Sequence, Type, TypeVar

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

        return object

    groups = defaultdict(list)
    for element in elements:
        element_type = get_type(key(element))
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
