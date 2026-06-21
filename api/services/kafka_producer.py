"""Kafka producer service for async message publishing."""
import json
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from api.config import settings

logger = structlog.get_logger()


class KafkaProducerService:
    """Async Kafka producer for publishing messages."""
    
    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.bootstrap_servers = settings.kafka_bootstrap_servers
        
    async def start(self):
        """Start Kafka producer."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type="gzip",
                acks='all',  # Wait for all replicas
            )
            await self.producer.start()
            logger.info("kafka_producer_started", servers=self.bootstrap_servers)
        except Exception as e:
            logger.error("kafka_producer_start_error", error=str(e))
            raise
    
    async def stop(self):
        """Stop Kafka producer."""
        if self.producer:
            await self.producer.stop()
            logger.info("kafka_producer_stopped")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """
        Send message to Kafka topic.
        
        Args:
            topic: Kafka topic name
            message: Message payload (will be JSON serialized)
            key: Optional message key for partitioning
            
        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            logger.error("kafka_producer_not_started")
            return False
        
        try:
            # Send message
            key_bytes = key.encode('utf-8') if key else None
            
            future = await self.producer.send(
                topic,
                value=message,
                key=key_bytes
            )
            
            # Wait for confirmation
            record_metadata = await future
            
            logger.info(
                "kafka_message_sent",
                topic=topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
                key=key
            )
            
            return True
            
        except KafkaError as e:
            logger.error(
                "kafka_send_error",
                topic=topic,
                error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "unexpected_kafka_error",
                topic=topic,
                error=str(e)
            )
            return False
    
    async def send_access_log(
        self,
        user_id: Optional[str],
        action: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Send access log to Kafka.
        
        Args:
            user_id: User identifier
            action: Action performed
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        from datetime import datetime
        
        message = {
            "user_id": user_id,
            "action": action,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message(
            topic=settings.kafka_topic_access_logs,
            message=message,
            key=user_id
        )
    
    async def send_event(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Send event to Kafka.
        
        Args:
            event_type: Type of event
            payload: Event payload
            
        Returns:
            True if successful
        """
        from datetime import datetime
        
        message = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message(
            topic=settings.kafka_topic_events,
            message=message
        )
    
    async def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert to Kafka.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity (info, warning, error, critical)
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        from datetime import datetime
        
        alert_message = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message(
            topic=settings.kafka_topic_alerts,
            message=alert_message
        )


# Global instance
kafka_producer = KafkaProducerService()
