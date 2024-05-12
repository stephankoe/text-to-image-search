"""Celery worker"""
import io
from typing import Iterable

from celery import Celery
from PIL import Image

from image_search.app.config import settings

app = Celery("image_search.app.tasks",
             backend=settings.worker.backend_url,
             broker=settings.worker.broker_url,
             broker_connection_retry_on_startup=True)


@app.task(result_expires=settings.worker.result_lifetime)
def index(images: Iterable[bytes],
          ) -> None:
    from image_search.app.initialize import database

    for image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        database.put(image)
