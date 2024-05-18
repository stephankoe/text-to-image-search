"""API"""
import base64
from typing import Optional

from celery.result import AsyncResult
from fastapi import FastAPI

from image_search.app.data import (IndexingJob, IndexRequest, SearchRequest,
                                   SearchResult, Trackable)
from image_search.app.initialize import database
from image_search.app.tasks import app as celery_app, index as index_task

app = FastAPI()


@app.post("/index")
def index(request: IndexRequest,
          ) -> IndexingJob:
    image_bytes = [base64.b64decode(image)
                   for image in request.images]
    task_result = index_task.delay(image_bytes)
    return IndexingJob(job_id=task_result.id,
                       status=task_result.status,
                       tracking_id=request.tracking_id)


@app.get("/index/{job_id}")
def query_indexing_job_status(job_id: str,
                              request: Optional[Trackable] = None,
                              ) -> IndexingJob:
    task_result = AsyncResult(job_id, app=celery_app)
    kwargs = {}
    if request:
        kwargs["tracking_id"] = request.tracking_id
    return IndexingJob(job_id=task_result.id,
                       status=task_result.status,
                       **kwargs)


@app.post("/search")
def search_images(request: SearchRequest,
                  ) -> SearchResult:
    candidates = database.query_similar(request.queries,
                                        n_similar=request.n_similar,
                                        )
    return SearchResult(queries=request.queries,
                        images=candidates,
                        tracking_id=request.tracking_id)
