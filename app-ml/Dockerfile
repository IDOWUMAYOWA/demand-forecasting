# This Docker file is for running ML Training and Inference Pipeline in Production
FROM python:3.11-slim

WORKDIR /app

# Install only production dependencies
COPY app-ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed to run the pipeline (not notebooks, not raw data)
COPY common/ ./common/
COPY config/ ./config/
COPY app-ml/src/ ./app-ml/src/
COPY app-ml/entrypoint/ ./app-ml/entrypoint/

ENV PYTHONPATH=/app:/app/app-ml/src

# Default command — docker-compose overrides this per service (train vs inference-api)
CMD ["python", "app-ml/entrypoint/train.py"]