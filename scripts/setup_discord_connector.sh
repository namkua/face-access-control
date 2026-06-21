#!/bin/bash

CONNECT_URL=${KAFKA_CONNECT_URL:-localhost:8083}
WEBHOOK_URL=${DISCORD_WEBHOOK_URL}

echo "Registering Discord HTTP Sink Connector via ${CONNECT_URL}..."

curl -X DELETE http://${CONNECT_URL}/connectors/discord-sink 2>/dev/null || true
sleep 2

curl -X POST http://${CONNECT_URL}/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "discord-sink",
    "config": {
      "connector.class": "io.aiven.kafka.connect.http.HttpSinkConnector",
      "tasks.max": "1",
      "topics": "alerts",
      "http.url": "'"${WEBHOOK_URL}"'",
      "http.headers.content.type": "application/json",
      "value.converter": "org.apache.kafka.connect.storage.StringConverter",
      "transforms": "WrapValue",
      "transforms.WrapValue.type": "org.apache.kafka.connect.transforms.HoistField$Value",
      "transforms.WrapValue.field": "content",
      "key.converter": "org.apache.kafka.connect.storage.StringConverter",
      "http.authorization.type": "none"
    }
  }'

echo ""
echo "Connector registered. Checking status:"
sleep 2
curl -s http://${CONNECT_URL}/connectors/discord-sink/status | jq
