FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
RUN set -x \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        language-pack-ja-base \
        language-pack-ja \
        language-pack-en \
        tzdata \
        sudo \
        tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN set -x \
    && update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja \
    && echo "$TZ" > /etc/timezone \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV LC_CTYPE C.UTF-8

RUN set -x \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
  && apt-get clean \
  && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN set -x \
  && pip install --no-cache-dir -U poetry==1.3.2 poetry-core==1.4.0

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR /root

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN set -x \
  && poetry export -f requirements.txt --with=dev -o requirements.txt \
  && pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir -U "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /workspace
