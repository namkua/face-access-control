FROM confluentinc/cp-kafka-connect:7.5.0

USER root

# Install required tools
RUN yum update -y && yum install -y wget tar && yum clean all

# Download and install Aiven http connector
RUN mkdir -p /usr/share/confluent-hub-components/aiven-kafka-connect-http && \
    wget https://github.com/Aiven-Open/http-connector-for-apache-kafka/releases/download/v0.9.0/http-connector-for-apache-kafka-0.9.0.tar -O /tmp/plugin.tar && \
    tar -xf /tmp/plugin.tar -C /usr/share/confluent-hub-components/aiven-kafka-connect-http --strip-components=1 && \
    rm /tmp/plugin.tar

# Install Confluent S3 Sink Connector
RUN confluent-hub install --no-prompt confluentinc/kafka-connect-s3:latest


USER appuser

