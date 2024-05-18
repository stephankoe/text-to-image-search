"""Initialize from arguments"""

from qdrant_client import QdrantClient

from image_search.app.config import settings
from image_search.core.database import Database, QdrantVectorDatabase
from image_search.core.embedding import CLIPEmbedder


def create_database(model_path: str,
                    device: str,
                    database_url: str,
                    collection_name: str,
                    ) -> Database:
    embed = CLIPEmbedder(model_path=model_path,
                         device=device,
                         )
    qdrant_client = QdrantClient(url=database_url)
    return QdrantVectorDatabase(embed=embed,
                                client=qdrant_client,
                                collection=collection_name,
                                )


database = create_database(settings.embedding.model_path,
                           settings.embedding.device,
                           settings.database.url,
                           settings.database.collection_name,
                           )
