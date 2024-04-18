"""API"""
import base64
from typing import Optional

from fastapi import FastAPI

from image_search.app.data import (IndexingJob, IndexingJobStatus, IndexRequest,
                                   SearchRequest, SearchResult, Trackable)
from image_search.app.initialize import database
from image_search.app.tasks import app as celery

app = FastAPI()


@app.post("/index")
def index(request: IndexRequest,
          ) -> IndexingJob:
    image_bytes = [base64.b64decode(image)
                   for image in request.images]
    celery.index(image_bytes)
    return IndexingJob(job_id='',  # TODO: job IDs
                       tracking_id=request.tracking_id)


@app.post("/index/{job_id}/status")
def query_indexing_job_status(request: Optional[Trackable] = None,
                              ) -> IndexingJobStatus:
    raise NotImplemented  # TODO


@app.post("/search")
def search_images(request: SearchRequest,
                  ) -> SearchResult:
    candidates = database.query_similar(request.queries,
                                        n_similar=request.n_similar,
                                        )
    return SearchResult(queries=request.queries,
                        images=candidates,
                        tracking_id=request.tracking_id)
