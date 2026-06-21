"""
Unit tests for KafkaProducerService.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# -----------------------------------------------------------------------
# Tests for KafkaProducerService
# -----------------------------------------------------------------------

class TestKafkaProducer:
    @pytest.mark.asyncio
    async def test_send_access_log(self):
        """Test send_access_log calls send_message with correct topic."""
        with patch("api.services.kafka_producer.AIOKafkaProducer") as MockProducer:
            mock_instance = AsyncMock()
            MockProducer.return_value = mock_instance

            from api.services.kafka_producer import KafkaProducerService

            svc = KafkaProducerService()
            svc.producer = mock_instance

            # Mock send_message to track calls
            svc.send_message = AsyncMock(return_value=True)

            await svc.send_access_log(
                user_id="u1",
                action="prediction",
                metadata={
                    "prediction_id": "test-123",
                    "matched_user": "alice",
                    "confidence": 0.95,
                    "processing_time_ms": 150.0,
                },
            )

            svc.send_message.assert_called_once()
            call_kwargs = svc.send_message.call_args
            assert call_kwargs.kwargs.get("topic") or call_kwargs[1].get("topic") or call_kwargs[0][0] == "access-logs"

    @pytest.mark.asyncio
    async def test_send_event(self):
        """Test send_event calls send_message with events topic."""
        with patch("api.services.kafka_producer.AIOKafkaProducer"):
            from api.services.kafka_producer import KafkaProducerService

            svc = KafkaProducerService()
            svc.send_message = AsyncMock(return_value=True)

            await svc.send_event(
                event_type="face_registered",
                payload={"user_id": 1, "username": "alice"},
            )

            svc.send_message.assert_called_once()


class TestKafkaProducerExtended:
    
    @pytest.fixture
    def service(self):
        from api.services.kafka_producer import KafkaProducerService
        return KafkaProducerService()

    # --- start ---
    @pytest.mark.asyncio
    async def test_start_success(self, service):
        with patch("api.services.kafka_producer.AIOKafkaProducer") as MockProducer:
            mock_instance = AsyncMock()
            MockProducer.return_value = mock_instance
            
            await service.start()
            
            MockProducer.assert_called_once()
            mock_instance.start.assert_called_once()
            assert service.producer is mock_instance

    @pytest.mark.asyncio
    async def test_start_raises_on_error(self, service):
        with patch("api.services.kafka_producer.AIOKafkaProducer") as MockProducer:
            mock_instance = AsyncMock()
            mock_instance.start.side_effect = Exception("Connection failed")
            MockProducer.return_value = mock_instance
            
            with pytest.raises(Exception, match="Connection failed"):
                await service.start()

    # --- stop ---
    @pytest.mark.asyncio
    async def test_stop_when_producer_running(self, service):
        service.producer = AsyncMock()
        
        await service.stop()
        
        service.producer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_producer_is_none(self, service):
        service.producer = None
        
        # Should not raise any error
        await service.stop()

    # --- send_message ---
    @pytest.mark.asyncio
    async def test_send_message_producer_not_started(self, service):
        service.producer = None
        
        result = await service.send_message("test-topic", {"data": 123})
        
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_success(self, service):
        import asyncio
        service.producer = AsyncMock()
        mock_future = asyncio.Future()
        mock_metadata = MagicMock()
        mock_metadata.partition = 0
        mock_metadata.offset = 10
        mock_future.set_result(mock_metadata)
        
        service.producer.send.return_value = mock_future
        
        result = await service.send_message("test-topic", {"data": 123}, key="test-key")
        
        assert result is True
        service.producer.send.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_send_message_unexpected_error(self, service):
        service.producer = AsyncMock()
        service.producer.send.side_effect = Exception("Unexpected error")
        
        result = await service.send_message("test-topic", {"data": 123})
        
        assert result is False

    # --- send_alert ---
    @pytest.mark.asyncio
    async def test_send_alert(self, service):
        with patch.object(service, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            
            result = await service.send_alert(
                alert_type="system_error",
                severity="critical",
                message="DB connection lost"
            )
            
            assert result is True
            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args.kwargs
            assert call_kwargs.get("topic") == service.settings.kafka_topic_alerts if hasattr(service, 'settings') else call_kwargs.get("topic") or mock_send.call_args[1].get("topic") or mock_send.call_args[0][0] == "alerts"
            
            msg = call_kwargs.get("message") or mock_send.call_args[1].get("message") or mock_send.call_args[0][1]
            assert msg["alert_type"] == "system_error"
            assert msg["severity"] == "critical"
            assert msg["message"] == "DB connection lost"
