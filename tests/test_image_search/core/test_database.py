"""Test database component"""
import os
import unittest
from unittest import mock
import uuid

from qdrant_client import QdrantClient
import torch

from image_search.core.database import QdrantVectorDatabase

_DEFAULT_QDRANT_URL = "localhost:6333"


class TestQdrantVectorDatabase(unittest.TestCase):

    def setUp(self):
        url = os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL)
        self._client = QdrantClient(url=url)
        self._collection = str(uuid.uuid4())
        self._embed = mock.MagicMock()
        self._database = QdrantVectorDatabase(embed=self._embed,
                                              client=self._client,
                                              collection=self._collection)

    def test_put_and_get(self):
        db_contents = {
            str(uuid.uuid4()): torch.rand(10,)
            for _ in range(10)
        }
        # TODO
