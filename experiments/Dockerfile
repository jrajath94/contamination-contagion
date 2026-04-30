FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git curl ca-certificates jq tini && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "transformers>=4.46,<5" \
        "peft>=0.13" \
        "trl>=0.11" \
        "datasets>=3.0" \
        "accelerate>=1.0" \
        "sentencepiece" \
        "hf_transfer" \
        "scikit-learn>=1.4" \
        "scipy" "pandas" "matplotlib" "seaborn"

COPY bootstrap_entry.sh /usr/local/bin/bootstrap_entry.sh
RUN chmod +x /usr/local/bin/bootstrap_entry.sh

ENV HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PAPER002_WORKDIR=/workspace/paper-002

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/bootstrap_entry.sh"]
