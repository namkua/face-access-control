FROM apache/airflow:2.7.3

USER root

# Install Java and other required dependencies for PySpark and OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre-headless \
        procps \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libpq-dev \
        gcc \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MinIO Client (mc)
RUN wget -q https://dl.min.io/client/mc/release/linux-amd64/mc \
    && chmod +x mc \
    && mv mc /usr/local/bin/mc

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

# Install Python dependencies needed by Airflow DAGs and PySpark jobs
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    apache-airflow-providers-apache-spark==4.3.0 \
    pyspark==3.4.1 \
    minio \
    kafka-python==2.0.2 \
    psycopg2-binary==2.9.9 \
    torch \
    torchvision \
    facenet-pytorch \
    opencv-python-headless \
    pillow==10.1.0 \
    requests \
    numpy
