#!/bin/bash

CONNECT_URL=${KAFKA_CONNECT_URL:-localhost:8083}
echo "Registering MinIO S3 Sink Connector via ${CONNECT_URL}..."

curl -X DELETE http://${CONNECT_URL}/connectors/minio-sink 2>/dev/null || true
sleep 2

curl -X POST http://${CONNECT_URL}/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "minio-sink",
    "config": {
      "connector.class": "io.confluent.connect.s3.S3SinkConnector",
      "tasks.max": "1",
      "topics": "access-logs",
      "s3.bucket.name": "access-logs-lake",
      "s3.region": "us-east-1",
      "s3.part.size": "5242880",
      "store.url": "http://minio:9000",
      "aws.access.key.id": "minioadmin",
      "aws.secret.access.key": "minioadmin",
      "storage.class": "io.confluent.connect.s3.storage.S3Storage",
      "format.class": "io.confluent.connect.s3.format.json.JsonFormat",
      "value.converter": "org.apache.kafka.connect.json.JsonConverter",
      "value.converter.schemas.enable": "false",
      "key.converter": "org.apache.kafka.connect.storage.StringConverter",
      "schema.compatibility": "NONE",
      "flush.size": "1000",
      "rotate.interval.ms": "60000",
      "rotate.schedule.interval.ms": "60000",
      "topics.dir": "",
      "partitioner.class": "io.confluent.connect.storage.partitioner.TimeBasedPartitioner",
      "path.format": "'\''Year='\''YYYY/'\''Month='\''MM/dd",
      "locale": "en",
      "timezone": "UTC",
      "partition.duration.ms": "3600000"
    }
  }'

echo ""
echo "Connector registered. Checking status:"
sleep 2
curl -s http://${CONNECT_URL}/connectors/minio-sink/status | jq
