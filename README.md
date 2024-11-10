# Text to Image Search

This service allows to search relevant images with a text prompt.  To this end, it uses a pretrained image-and-text encoder model, such as [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) to encode images and text queries into an n-dimensional vector space.  The service then returns the images whose embeddings are closest to the text query's embedding, as measured with the cosine similarity.


## Usage

### Docker

- Build the image: 
  ```bash
  docker compose build
  ```
- Start the service(s): 
  ```bash
  docker compose up
  ```

### Python

- Install project requirements using
  ```bash
  pip install -r requirements.txt
  ```
- Prepare image data
- Start vector database, e.g., with Docker:
  ```bash
  docker run -p 6333:6333 --mount type=bind,source=$HOME/.qdrant_data,target=/qdrant_data --name vector-db qdrant/qdrant:v1.8.4
  ```
- Start encoding task queue
  ```bash
  bash deploy/entrypoint.sh task-queue
  ```
- Start HTTP server
  ```bash
  bash deploy/entrypoint.sh server
  ```
