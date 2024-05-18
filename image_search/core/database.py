import base64
from typing import Iterable, Protocol, Sequence, Tuple, TypeVar
import uuid

from PIL.Image import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, PointStruct, ScoredPoint,
                                  SearchRequest, VectorParams)

from image_search.core.embedding import Embedder
from image_search.core.utils import scale_down

_T = TypeVar("_T")


class Database(Protocol):

    def put(self,
            obj: _T | Iterable[_T],
            ) -> None:
        """
        Put object into database
        :param obj: object to store
        """
        ...

    def query_similar(self,
                      obj: _T | Iterable[_T],
                      n_similar: int = None,
                      ) -> Iterable[Iterable[_T]]:
        """
        Get similar objects from database
        :param obj: reference object(s) for query
        :param n_similar: number of candidates
        :return: similar objects from database
        """
        ...


class QdrantVectorDatabase:

    _OBJ_TYPE = str | Image

    def __init__(self,
                 embed: Embedder,
                 client: QdrantClient,
                 collection: str = None,
                 thumbnail_size: Tuple[int, int] = (128, 128),
                 ):
        """
        :param embed: object embedding function
        :param client: Qdrant client object
        :param collection: already existing collection to use
        :param thumbnail_size: size of thumbnail that is stored in DB
        """
        self._embed = embed
        self._client = client
        self._thumbnail_size = thumbnail_size
        if not collection or not self._client.collection_exists(collection):
            self._collection = self._initialize_collection(collection)
        else:
            self._collection = collection

    def put(self,
            objs: _OBJ_TYPE | Iterable[_OBJ_TYPE],
            ) -> None:
        """
        Put objects into Qdrant database
        :param objs: object(s) to store
        """
        if isinstance(objs, str | Image):
            objs = [objs]

        embeddings = self._embed(objs)
        payloads = (
            self._text_to_payload(obj)
            if isinstance(obj, str)
            else self._image_to_payload(obj)
            for obj in objs
        )

        points = [
            PointStruct(id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=payload,
                        )
            for embedding, payload in zip(embeddings, payloads)
        ]
        # TODO: check result
        # TODO: async?
        self._client.upsert(collection_name=self._collection,
                            points=points,
                            )

    def query_similar(self,
                      objs: _OBJ_TYPE | Iterable[_OBJ_TYPE],
                      n_similar: int = 5,
                      ) -> Iterable[Iterable[_OBJ_TYPE]]:
        """
        Get similar texts or images from the database
        :param objs: reference text(s) or image(s)
        :param n_similar: number of similar objects to return
        :return: similar objects from database
        """
        if isinstance(objs, str | Image):
            objs = [objs]

        embeddings = self._embed(objs)
        requests = [
            SearchRequest(vector=embedding.tolist(),
                          limit=n_similar,
                          )
            for embedding in embeddings
        ]

        # TODO: async?
        hits = self._client.search_batch(collection_name=self._collection,
                                         requests=requests,
                                         )
        return self._extract_payloads(hits)

    def _image_to_payload(self,
                          image: Image,
                          ) -> dict[str, bytes]:
        """
        Transform image to payload
        :param image: image
        :return: payload dict for database
        """
        image.thumbnail(scale_down(image.size, self._thumbnail_size))
        return {
            "pixels": base64.b64encode(image.tobytes()),
        }

    def _initialize_collection(self,
                               collection_name: str = None,
                               ) -> str:
        """
        Initialize collection and return collection name
        :param embed: embedding function
        :param collection_name: name of collection (optional)
        :return: name of initialized collection
        """
        collection_name = collection_name or str(uuid.uuid4())
        size = self._embed.embedding_dim
        distance = Distance(self._embed.distance.title())
        config = VectorParams(size=size,
                              distance=distance)
        self._client.recreate_collection(collection_name=collection_name,
                                         vectors_config=config)
        return collection_name

    @staticmethod
    def _extract_payloads(results: Sequence[Sequence[ScoredPoint]],
                          ) -> Sequence[Sequence[str | bytes]]:
        return [
            [
                scored_point.payload["text"]
                if "text" in scored_point.payload
                else scored_point.payload["pixels"]
                for scored_point in candidates
            ]
            for candidates in results
        ]

    @staticmethod
    def _text_to_payload(text: str,
                         ) -> dict[str, str]:
        """
        Transform text to payload
        :param text: text
        :return: payload dict for database
        """
        return {
            "text": text,
        }
