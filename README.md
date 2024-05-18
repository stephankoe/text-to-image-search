# Text to Image Search

Search relevant images based on a text input

## Usage

### Docker

- Build the image: docker compose build
- Start the service(s): docker compose up
- how to mount data

### Python

- install requirements
- download/prepare data
- start database
- start server


## Dataset analysis

### Ad Image Dataset

Dataset description:

  - Contains images without description
  - Advertisement domain

Common characteristics:

  - Many/most images contain chunks of text
  - Brand names are important
  - Brand symbols

Possible queries:

  - Queries for objects (car, computer, ...)
  - Queries for brands
  - Queries for products
  - Queries for product categories

## Architecture

- architectural overview
  - api
  - encoder
  - retriever
  - indexer
- techniques involved
  - server: probably gunicorn, torch is blocking
  - huggingface img classification model

## Challenges

- challenges encountered during implementation

1. Create collections yields
   ```
   qdrant_client.http.exceptions.ResponseHandlingException: illegal request line
   ```
   Solved by using port 6333 instead of 6334
2. When calling the model within a Celery worker, the process was hanging
   indefinitely when calling the model. 
   Solution: initialize the model inside the task function.

## Improvements

- what could be improved in the future
- usability enhancements
- security
- backend for storing images

## TODO

- [ ] Indexing task queue
- [ ] Model server (torchserve)
- [ ] Unblocking code
- [ ] Proper configuration
- [ ] Proper logging
- [ ] Architecture description (try out the C4 model)
- [ ] Finish this document
- [ ] Experimental evaluation on 2-3 datasets and find badcases
- [ ] Try out 3-4 different models
- [ ] Remove ports from Dockerfile
- [ ] Test cases (end-to-end with HTTP queries, unit tests)
- [ ] Docker