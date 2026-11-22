ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS dependencies
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 LUMENSTRIP_DATA=/var/lib/lumenstrip OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /opt/lumenstrip
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY lumenstrip lumenstrip
COPY static static
RUN useradd -U -u 11601 lumenstrip && install -d -o lumenstrip -g lumenstrip /var/lib/lumenstrip
USER lumenstrip
EXPOSE 8000
CMD ["python", "-m", "lumenstrip"]
