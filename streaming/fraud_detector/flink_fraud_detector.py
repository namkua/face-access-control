"""
Apache Flink job for real-time fraud detection.

Detects:
1. Spam: Too many successful requests from a single user within a time window.
2. Brute-force: Too many 'no_match' predictions from a single IP within a time window.

Uses Flink KeyedProcessFunction with ListState and Event Time Watermarks for production-grade state management.
"""
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import (
    KafkaSource, KafkaOffsetsInitializer,
    KafkaSink, KafkaRecordSerializationSchema
)
from pyflink.datastream.functions import KeyedProcessFunction, MapFunction
from pyflink.datastream.state import ListStateDescriptor
from pyflink.common.typeinfo import Types
from pyflink.common.time import Duration
from pyflink.common.watermark_strategy import TimestampAssigner
import json
import time
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value, record_timestamp):
        try:
            ts_str = value.get('timestamp')
            if ts_str:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
        except Exception as e:
            logger.debug(f"Error parsing timestamp: {e}")
        return int(time.time() * 1000)


class ParseJsonMapFunction(MapFunction):
    def map(self, value):
        try:
            return json.loads(value)
        except Exception:
            return {}


class SpamDetectionFunction(KeyedProcessFunction):
    """Detect spam: too many successful logins from the same authenticated user in a time window."""

    def __init__(self, threshold: int, window_seconds: int):
        self.threshold = threshold
        self.window_seconds = window_seconds

    def open(self, runtime_context):
        descriptor = ListStateDescriptor("spam_events", Types.LONG())
        self.events_state = runtime_context.get_list_state(descriptor)

    def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
        user_id = value.get('user_id', 'anonymous')
        metadata = value.get('metadata', {})
        prediction_status = metadata.get('prediction_status', '')

        if user_id == 'anonymous' or prediction_status != 'success':
            return

        current_timestamp = ctx.timestamp()
        if current_timestamp is None:
            return
            
        window_start = current_timestamp - self.window_seconds * 1000

        # Read existing events, filter, and count
        events = []
        for ts in self.events_state.get():
            if ts > window_start:
                events.append(ts)
        
        events.append(current_timestamp)

        # Update state with filtered events
        self.events_state.clear()
        self.events_state.add_all(events)

        count = len(events)
        if count > self.threshold:
            source_ip = metadata.get('source_ip', 'unknown')
            alert = {
                'alert_type': 'spam_detection',
                'user_id': user_id,
                'source_ip': source_ip,
                'count': count,
                'window_seconds': self.window_seconds,
                'severity': 'critical' if count > self.threshold * 2 else 'warning',
                'message': f'User {user_id} logged in successfully {count} times in {self.window_seconds}s',
                'timestamp': value.get('timestamp', '')
            }
            logger.warning(f"SPAM DETECTED: {alert}")
            yield json.dumps(alert)

        # Register a timer to clean up state
        ctx.timer_service().register_event_time_timer(current_timestamp + self.window_seconds * 1000 + 1)

    def on_timer(self, timestamp: int, ctx: 'KeyedProcessFunction.OnTimerContext'):
        window_start = timestamp - self.window_seconds * 1000
        events = []
        for ts in self.events_state.get():
            if ts > window_start:
                events.append(ts)
        
        self.events_state.clear()
        if events:
            self.events_state.add_all(events)


class BruteForceDetectionFunction(KeyedProcessFunction):
    """Detect brute-force: too many 'no_match' predictions from the same IP."""

    def __init__(self, threshold: int, window_seconds: int):
        self.threshold = threshold
        self.window_seconds = window_seconds

    def open(self, runtime_context):
        descriptor = ListStateDescriptor("bf_events", Types.LONG())
        self.events_state = runtime_context.get_list_state(descriptor)

    def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
        action = value.get('action')
        metadata = value.get('metadata', {})

        if action != 'prediction' or metadata.get('prediction_status') != 'no_match':
            return

        current_timestamp = ctx.timestamp()
        if current_timestamp is None:
            return

        window_start = current_timestamp - self.window_seconds * 1000

        events = []
        for ts in self.events_state.get():
            if ts > window_start:
                events.append(ts)
        
        events.append(current_timestamp)

        self.events_state.clear()
        self.events_state.add_all(events)

        count = len(events)
        if count > self.threshold:
            source_ip = metadata.get('source_ip', 'unknown')
            alert = {
                'alert_type': 'brute_force_detection',
                'source_ip': source_ip,
                'count': count,
                'window_seconds': self.window_seconds,
                'severity': 'critical',
                'message': f'IP {source_ip} had {count} failed matches in {self.window_seconds}s',
                'timestamp': value.get('timestamp', '')
            }
            logger.warning(f"BRUTE FORCE DETECTED: {alert}")
            yield json.dumps(alert)

        ctx.timer_service().register_event_time_timer(current_timestamp + self.window_seconds * 1000 + 1)

    def on_timer(self, timestamp: int, ctx: 'KeyedProcessFunction.OnTimerContext'):
        window_start = timestamp - self.window_seconds * 1000
        events = []
        for ts in self.events_state.get():
            if ts > window_start:
                events.append(ts)
        
        self.events_state.clear()
        if events:
            self.events_state.add_all(events)


def create_fraud_detection_job():
    """Create and configure the Flink fraud detection job."""

    env = StreamExecutionEnvironment.get_execution_environment()
    
    # We can now use parallelism > 1 as state is keyed!
    # Set parallelism dynamically from environment, defaulting to 1 for local docker testing
    parallelism = int(os.getenv("FLINK_PARALLELISM", "1"))
    env.set_parallelism(parallelism)
    
    env.enable_checkpointing(10000)
    env.add_jars("file:///opt/flink/lib/flink-sql-connector-kafka-3.0.1-1.18.jar")

    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "access-logs")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "alerts")

    spam_threshold = int(os.getenv("SPAM_THRESHOLD", "5"))
    spam_window = int(os.getenv("SPAM_WINDOW_SECONDS", "60"))

    bf_threshold = int(os.getenv("BRUTE_FORCE_THRESHOLD", "10"))
    bf_window = int(os.getenv("BRUTE_FORCE_WINDOW_SECONDS", "300"))

    logger.info(f"Config: kafka={kafka_servers}, input={input_topic}, output={output_topic}")
    logger.info(f"Spam: threshold={spam_threshold}, window={spam_window}s")
    logger.info(f"BruteForce: threshold={bf_threshold}, window={bf_window}s")

    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers(kafka_servers) \
        .set_topics(input_topic) \
        .set_group_id("fraud-detector") \
        .set_starting_offsets(KafkaOffsetsInitializer.latest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    raw_stream = env.from_source(
        kafka_source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Access Logs Source"
    )

    # 1. Parse JSON
    parsed_stream = raw_stream.map(
        ParseJsonMapFunction(), 
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    # 2. Assign Timestamps and Watermarks
    watermark_strategy = WatermarkStrategy \
        .for_bounded_out_of_orderness(Duration.of_seconds(5)) \
        .with_timestamp_assigner(CustomTimestampAssigner())
        
    watermarked_stream = parsed_stream.assign_timestamps_and_watermarks(watermark_strategy)

    # 3. Branch 1: Spam Detection (Keyed by user_id)
    spam_alerts = watermarked_stream \
        .key_by(lambda x: x.get('user_id', 'anonymous')) \
        .process(
            SpamDetectionFunction(threshold=spam_threshold, window_seconds=spam_window),
            output_type=Types.STRING()
        )

    # 4. Branch 2: Brute Force Detection (Keyed by source_ip)
    bf_alerts = watermarked_stream \
        .key_by(lambda x: x.get('metadata', {}).get('source_ip', 'unknown')) \
        .process(
            BruteForceDetectionFunction(threshold=bf_threshold, window_seconds=bf_window),
            output_type=Types.STRING()
        )

    # 5. Merge all alerts
    all_alerts = spam_alerts.union(bf_alerts)

    kafka_sink = KafkaSink.builder() \
        .set_bootstrap_servers(kafka_servers) \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
                .set_topic(output_topic)
                .set_value_serialization_schema(SimpleStringSchema())
                .build()
        ) \
        .build()

    all_alerts.sink_to(kafka_sink)
    all_alerts.print() 

    env.execute("Fraud Detection Job")


if __name__ == "__main__":
    logger.info("Starting Flink Fraud Detection Job...")
    create_fraud_detection_job()
