"""Test embedding component"""
import io
import os
import unittest

from PIL import Image
import torch

import image_search
from image_search.core.embedding import CLIPEmbedder
from image_search.core.utils import cosine_similarity

_DEFAULT_MODEL_PATH = "openai/clip-vit-base-patch32"
_ROOT = os.path.dirname(os.path.dirname(image_search.__file__))
_TEST_ROOT = os.path.join(_ROOT, "tests")
_FIXTURES_PATH = os.path.join(_TEST_ROOT, "fixtures")


class TestCLIPEmbedder(unittest.TestCase):

    def setUp(self):
        model_path = os.getenv("CLIP_MODEL_PATH", _DEFAULT_MODEL_PATH)
        self._embed = CLIPEmbedder(model_path)

    def test_call__texts(self):
        embeddings = self._embed(["The quick brown fox jumps over the lazy dog",
                                  "Delicious hot dogs",
                                  "Dog",
                                  "Yellow submarine",
                                  ])
        embeddings = torch.stack(tuple(embeddings))
        similarities = cosine_similarity(embeddings)
        self.assertLess(similarities[0, 1].item(), similarities[0, 2].item())
        self.assertLess(similarities[0, 3].item(), similarities[0, 1].item())

    def test_call__text_images(self):
        dog_image = Image.open(os.path.join(_FIXTURES_PATH, "dog.jpg"))
        cat_image = Image.open(os.path.join(_FIXTURES_PATH, "cat.jpg"))
        embeddings = self._embed(["Cute dog standing on two legs",
                                  dog_image,
                                  cat_image,
                                  ])
        embeddings = torch.stack(tuple(embeddings))
        similarities = cosine_similarity(embeddings)
        self.assertLess(similarities[0, 2].item(), similarities[0, 1].item())

    def test_call__images_from_memory(self):
        dog_image = os.path.join(_FIXTURES_PATH, "dog.jpg")
        with open(dog_image, 'rb') as fh:
            image_bytes = fh.read()

        image = Image.open(io.BytesIO(image_bytes))
        embeddings = self._embed([image])
        self.assertIsNotNone(embeddings)
