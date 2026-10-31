# Three images from one file:
#   docker build --target api    -t alphastream-api .
#   docker build --target ui     -t alphastream-ui .
#   docker build --target mlflow -t alphastream-mlflow .

FROM python:3.12.7-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app


FROM base AS api
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY configs ./configs
COPY src ./src
COPY train.py backtest.py ./
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p models results data .cache && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS ui
COPY requirements-ui.txt ./
RUN pip install -r requirements-ui.txt
COPY ui ./ui
COPY .streamlit ./.streamlit
USER app
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]


FROM base AS mlflow
RUN pip install mlflow==3.17.2 && mkdir -p /mlflow && chown app:app /mlflow
USER app
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "*"]
