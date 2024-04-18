FROM python:3.11-slim

COPY requirements.txt /tmp/

RUN apt update && \
    apt install -y iproute2 && \
    apt clean && \
    groupadd --gid 1000 service && \
    useradd --gid 1000 --uid 1000 -ms /bin/bash service && \
    pip install -r requirements.txt

COPY . /opt/cloud/image-search

RUN find /opt/cloud/image-search -type d -exec chmod 500 {} + \
    find /opt/cloud/image-search -type f -exec chmod 400 {} + \
    find /opt/cloud/image-search -exec chown service:service {} + \
    umask 277 \

USER service
EXPOSE 8080
ENTRYPOINT /opt/cloud/deploy/entrypoint.sh