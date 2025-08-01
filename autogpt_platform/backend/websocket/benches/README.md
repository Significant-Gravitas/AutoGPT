# WebSocket Server Benchmarks

This directory contains performance benchmarks for the AutoGPT WebSocket server.

## Prerequisites

1. Redis must be running locally or set `REDIS_URL` environment variable:
   ```bash
   docker run -d -p 6379:6379 redis:latest
   ```

2. Build the project in release mode:
   ```bash
   cargo build --release
   ```

## Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark group:
```bash
cargo bench connection_establishment
cargo bench subscriptions
cargo bench message_throughput
cargo bench concurrent_connections
cargo bench message_parsing
cargo bench redis_event_processing
```

## Benchmark Categories

### Connection Establishment
Tests the performance of establishing WebSocket connections with different authentication scenarios:
- No authentication
- Valid JWT authentication
- Invalid JWT authentication (connection rejection)

### Subscriptions
Measures the performance of subscription operations:
- Subscribing to graph execution events
- Unsubscribing from channels

### Message Throughput
Tests how many messages the server can process per second with varying message counts (10, 100, 1000).

### Concurrent Connections
Benchmarks the server's ability to handle multiple simultaneous connections (10, 50, 100, 500 clients).

### Message Parsing
Tests JSON parsing performance with different message sizes (100B to 100KB).

### Redis Event Processing
Benchmarks the parsing of execution events received from Redis.

## Profiling

To generate flamegraphs for CPU profiling:

1. Install flamegraph tools:
   ```bash
   cargo install flamegraph
   ```

2. Run benchmarks with profiling:
   ```bash
   cargo bench --bench websocket_bench -- --profile-time=10
   ```

## Interpreting Results

- **Throughput**: Higher is better (operations/second or elements/second)
- **Time**: Lower is better (nanoseconds per operation)
- **Error margins**: Look for stable results with low standard deviation

## Optimizing Performance

Based on benchmark results, consider:

1. **Connection pooling** for Redis connections
2. **Message batching** for high-throughput scenarios
3. **Async task tuning** for concurrent connection handling
4. **JSON parsing optimization** using simd-json or other fast parsers
5. **Memory allocation** optimization using arena allocators

## Notes

- Benchmarks create actual WebSocket servers on random ports
- Each benchmark iteration properly cleans up resources
- Results may vary based on system resources and Redis performance