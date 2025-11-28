FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg libsm6 libxext6

WORKDIR /app
COPY requirements.txt ./
RUN pip install --trusted-host https://mirror-pypi.runflare.com -i https://mirror-pypi.runflare.com/simple/ --no-cache-dir -r requirements.txt
RUN mkdir -p ./data/

COPY main.py object_tracking.py line_crossing.py configs.yaml bytetrack.yaml ./

CMD ["python3", "main.py"]