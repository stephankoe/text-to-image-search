FROM python:3.11-slim

ARG pip_opts

COPY requirements.txt /tmp/

RUN apt-get update && \
    apt-get install -y iproute2 && \
    apt-get clean && \
    groupadd --gid 1000 service && \
    useradd --gid 1000 --uid 1000 -ms /bin/bash service && \
    pip install --no-cache-dir $pip_opts -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

COPY .. /opt/cloud/image-search

WORKDIR /opt/cloud/image-search

RUN find . -type d -exec chmod 500 {} + && \
    find . -type f -exec chmod 400 {} + && \
    find . -exec chown service:service {} + && \
    umask 277

USER service
EXPOSE 8080
ENTRYPOINT ["bash", "/opt/cloud/image-search/deploy/entrypoint.sh"]
CMD ["server"]