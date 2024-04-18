"""Data transfer objects"""
from typing import List, Optional

from pydantic import BaseModel


class Trackable(BaseModel):
    """Adds a tracking ID to track requests through multiple microservices"""

    tracking_id: Optional[int] = None


class SearchRequest(Trackable):
    """Search request body"""

    queries: List[str]  # user queries
    n_similar: int = 5  # number of candidate images per query


class SearchResult(Trackable):
    """Search result body"""

    queries: List[str]  # user queries
    images: List[List[str]]  # resulting images


class IndexRequest(Trackable):
    """Index request object"""

    images: List[str]  # base64 encoded image data


class IndexingJob(Trackable):
    """Indexing job information"""

    job_id: str


class IndexingJobStatus(Trackable):
    """Indexing job status"""

    job_id: str
    status: str
