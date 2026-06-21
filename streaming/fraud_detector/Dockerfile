FROM flink:1.18-scala_2.12-java11

# Install Python 3 and JDK (needed to build pemja on arm64)
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev openjdk-11-jdk-headless && \
    ln -s /usr/lib/jvm/java-11-openjdk-*/include /opt/java/openjdk/include || true && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Download Kafka connector JARs
RUN mkdir -p /opt/flink/lib && \
    wget -P /opt/flink/lib/ https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.0.1-1.18/flink-sql-connector-kafka-3.0.1-1.18.jar

# Install Python dependencies
COPY requirements.txt /opt/flink/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/flink/requirements.txt

# Copy job script
COPY flink_fraud_detector.py /opt/flink/usrlib/

# Command to submit the job (can be overridden in docker-compose)
CMD ["flink", "run", "-py", "/opt/flink/usrlib/flink_fraud_detector.py"]
