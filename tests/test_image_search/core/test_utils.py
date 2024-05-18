from operator import itemgetter
import unittest

import torch

from image_search.core.utils import cosine_similarity, create_batches, \
    group_by_type, scale_down


class TestGroupByType(unittest.TestCase):

    def test_group_by_type__default(self):
        elements = [1, 2, None, "one", 3, None, "two"]
        expected = {
            int: [1, 2, 3],
            type(None): [None, None],
            str: ["one", "two"],
        }
        output = group_by_type(elements)
        self.assertEquals(expected, output)

    def test_group_by_type__with_key(self):
        elements = enumerate([1, 2, None, "one", 3, None, "two"])
        expected = {
            int: [(0, 1), (1, 2), (4, 3)],
            type(None): [(2, None), (5, None)],
            str: [(3, "one"), (6, "two")],
        }
        output = group_by_type(elements, key=itemgetter(1))
        self.assertEquals(expected, output)

    def test_group_by_type__with_filter(self):
        elements = [1, 2, None, "one", 3, None, "two"]
        expected = {
            int: [1, 2, 3],
            str: ["one", "two"],
        }
        output = group_by_type(elements, types={str, int})
        self.assertEquals(expected, output)


class TestCosineSimilarity(unittest.TestCase):

    def test_cosine_similarity(self):
        vectors = torch.tensor([[0.6891, 0.2454, 0.7934, 0.1928, 0.2603],
                                [0.6270, 0.3612, 0.9891, 0.5631, 0.5761],
                                [0.7357, 0.3241, 0.0337, 0.8078, 0.5221],
                                [0.8073, 0.5028, 0.7864, 0.8364, 0.8515],
                                [0.9753, 0.1112, 0.6858, 0.6813, 0.3945],
                                [0.3479, 0.7076, 0.2716, 0.6678, 0.8997],
                                [0.3414, 0.3199, 0.5121, 0.0189, 0.8661]])
        expected = torch.tensor([[1.0000, 0.9465, 0.6404, 0.8719, 0.9151, 0.6302, 0.7591],
                                 [0.9465, 1.0000, 0.7435, 0.9639, 0.9235, 0.7990, 0.8268],
                                 [0.6404, 0.7435, 1.0000, 0.8842, 0.8530, 0.8584, 0.6033],
                                 [0.8719, 0.9639, 0.8842, 1.0000, 0.9301, 0.9069, 0.8358],
                                 [0.9151, 0.9235, 0.8530, 0.9301, 1.0000, 0.7065, 0.6755],
                                 [0.6302, 0.7990, 0.8584, 0.9069, 0.7065, 1.0000, 0.8232],
                                 [0.7591, 0.8268, 0.6033, 0.8358, 0.6755, 0.8232, 1.0000]])
        output = cosine_similarity(vectors)
        torch.testing.assert_allclose(output, expected)


class TestCreateBatches(unittest.TestCase):

    def test_create_batches__empty(self):
        batches = list(create_batches([], batch_size=5))
        self.assertEquals([], batches)

    def test_create_batches__even(self):
        elements = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        batches = list(create_batches(elements, batch_size=5))
        expected = [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9),
                    (10, 11, 12, 13, 14), (15, 16, 17, 18, 19),
                    (20, 21, 22, 23, 24), (25, 26, 27, 28, 29),
                    (30, 31, 32, 33, 34), (35, 36, 37, 38, 39),
                    (40, 41, 42, 43, 44), (45, 46, 47, 48, 49)]
        self.assertEquals(expected, batches)

    def test_create_batches__odd(self):
        elements = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        batches = list(create_batches(elements, batch_size=16))
        expected = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15), (16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31), (32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47), (48, 49)]
        self.assertEquals(expected, batches)


class TestScaleDown(unittest.TestCase):

    def test_scale_down__larger(self):
        result = scale_down(current_size=(500, 300),
                            target_size=(100, 100))
        expected = (100, 60)
        self.assertEquals(expected, result)

    def test_scale_down__smaller(self):
        result = scale_down(current_size=(100, 100),
                            target_size=(500, 300))
        expected = (100, 100)
        self.assertEquals(expected, result)

    def test_scale_down__zero(self):
        result = scale_down(current_size=(100, 300),
                            target_size=(0, 0))
        expected = (0, 0)
        self.assertEquals(expected, result)

    def test_scale_down__rounded(self):
        result = scale_down(current_size=(333, 123),
                            target_size=(100, 100))
        expected = (100, 36)
        self.assertEquals(expected, result)
