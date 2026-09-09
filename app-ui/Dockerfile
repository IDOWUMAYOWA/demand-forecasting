# Dockerfile for running the Dash UI in production
FROM python:3.11-slim

WORKDIR /app

COPY app-ui/requirements.txt ./app-ui/requirements.txt
RUN pip install --no-cache-dir -r app-ui/requirements.txt

COPY common/ ./common/
COPY config/ ./config/
COPY app-ui/ ./app-ui/

# gunicorn needs to import app.py as a plain module — "app-ui" has a
# hyphen, which isn't a valid Python package name for dotted imports —
# so we run from inside app-ui/ instead of referencing it as a package.
WORKDIR /app/app-ui

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]