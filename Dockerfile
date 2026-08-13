# "pipeline" onde definimos os steps que seguem um fluxo logico para isolar - isolamento de especificicações de um modelo operacional, o container roda sempre em cima do host, ou seja, os rescursos da maquina host, são utilizados no container docker

# Stage 1
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv export --format requirements-txt --output-file requirements.txt --no-dev


# Stage 2
FROM python:3.11.15-alpine AS final

WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY /app/api.py ./
EXPOSE 8000
CMD ["uvicorn","api:app", "--host", "0.0.0.0", "--port", "8000"]


# esta dando erro porque a image alpine n tem alguns compiladores do sklearn - precisa arrumar