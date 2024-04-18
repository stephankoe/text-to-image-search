"""Celery worker"""
from typing import Iterable

from celery import Celery

from image_search.app.initialize import database
from image_search.app.config import settings

app = Celery("image_search.app.tasks", broker=settings.worker.broker_url)


@app.task
def index(images: Iterable[bytes],
          ) -> None:
    for image in images:
        database.put(image)
