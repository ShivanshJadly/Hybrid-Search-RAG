FROM python:3.12-slim

# Build tools are not needed: chromadb>=1.0 ships binary wheels, and torch is
# pulled from the CPU index. Keeping this image compiler-free keeps it small.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"

COPY src ./src
RUN pip install -e .

# Bake the models into the image so the container does not download ~200MB on
# first request.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/healthz',timeout=4).status_code==200 else 1)"

CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
