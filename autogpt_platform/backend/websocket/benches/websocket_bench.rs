#![allow(clippy::unwrap_used)] // Benchmarks can panic on setup errors

use axum::{routing::get, Router};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio_tungstenite::{connect_async, tungstenite::Message};

// Import the actual websocket server components
use websocket::{models, ws_handler, AppState, Config, ConnectionManager, Stats};

// Helper to create a test server
async fn create_test_server(enable_auth: bool) -> (String, tokio::task::JoinHandle<()>) {
    // Set environment variables for test config
    std::env::set_var("WEBSOCKET_SERVER_HOST", "127.0.0.1");
    std::env::set_var("WEBSOCKET_SERVER_PORT", "0");
    std::env::set_var("ENABLE_AUTH", enable_auth.to_string());
    std::env::set_var("SUPABASE_JWT_SECRET", "test_secret");
    std::env::set_var("DEFAULT_USER_ID", "test_user");
    if std::env::var("REDIS_URL").is_err() {
        std::env::set_var("REDIS_URL", "redis://localhost:6379");
    }

    let mut config = Config::load(None);
    config.port = 0; // Force OS to assign port

    let redis_client =
        redis::Client::open(config.redis_url.clone()).expect("Failed to connect to Redis");
    let stats = Arc::new(Stats::default());
    let mgr = Arc::new(ConnectionManager::new(
        redis_client,
        config.execution_event_bus_name.clone(),
        stats.clone(),
    ));

    // Start broadcaster
    let mgr_clone = mgr.clone();
    tokio::spawn(async move {
        mgr_clone.run_broadcaster().await;
    });

    let state = AppState {
        mgr,
        config: Arc::new(config),
        stats,
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(axum::Extension(state));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_url = format!("ws://{addr}");

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    (server_url, server_handle)
}

// Helper to create a valid JWT token
fn create_jwt_token(user_id: &str) -> String {
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    use serde::Serialize;

    #[derive(Serialize)]
    struct Claims {
        sub: String,
        aud: Vec<String>,
        exp: usize,
    }

    let claims = Claims {
        sub: user_id.to_string(),
        aud: vec!["authenticated".to_string()],
        exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp() as usize,
    };

    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(b"test_secret"),
    )
    .unwrap()
}

// Benchmark connection establishment
fn benchmark_connection_establishment(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("connection_establishment");
    group.measurement_time(Duration::from_secs(30));

    // Test without auth
    group.bench_function("no_auth", |b| {
        b.to_async(&rt).iter_with_large_drop(|| async {
            let (server_url, server_handle) = create_test_server(false).await;
            let url = format!("{server_url}/ws");
            let (ws_stream, _) = connect_async(&url).await.unwrap();
            drop(ws_stream);
            server_handle.abort();
        });
    });

    // Test with valid auth
    group.bench_function("valid_auth", |b| {
        b.to_async(&rt).iter_with_large_drop(|| async {
            let (server_url, server_handle) = create_test_server(true).await;
            let token = create_jwt_token("test_user");
            let url = format!("{server_url}/ws?token={token}");
            let (ws_stream, _) = connect_async(&url).await.unwrap();
            drop(ws_stream);
            server_handle.abort();
        });
    });

    // Test with invalid auth
    group.bench_function("invalid_auth", |b| {
        b.to_async(&rt).iter_with_large_drop(|| async {
            let (server_url, server_handle) = create_test_server(true).await;
            let url = format!("{server_url}/ws?token=invalid");
            let result = connect_async(&url).await;
            assert!(
                result.is_err() || {
                    if let Ok((mut ws_stream, _)) = result {
                        // Should receive close frame
                        matches!(ws_stream.next().await, Some(Ok(Message::Close(_))))
                    } else {
                        false
                    }
                }
            );
            server_handle.abort();
        });
    });

    group.finish();
}

// Benchmark subscription operations
fn benchmark_subscriptions(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("subscriptions");
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("subscribe_graph_execution", |b| {
        b.to_async(&rt).iter_with_large_drop(|| async {
            let (server_url, server_handle) = create_test_server(false).await;
            let url = format!("{server_url}/ws");
            let (mut ws_stream, _) = connect_async(&url).await.unwrap();
            let msg = json!({
                "method": "subscribe_graph_execution",
                "data": {
                    "graph_exec_id": "test_exec_123"
                }
            });

            ws_stream
                .send(Message::Text(msg.to_string()))
                .await
                .unwrap();

            // Wait for response
            if let Some(Ok(Message::Text(response))) = ws_stream.next().await {
                let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
                assert_eq!(resp["success"], true);
            }

            server_handle.abort();
        });
    });

    group.bench_function("unsubscribe", |b| {
        b.to_async(&rt).iter_with_large_drop(|| async {
            let (server_url, server_handle) = create_test_server(false).await;
            let url = format!("{server_url}/ws");
            let (mut ws_stream, _) = connect_async(&url).await.unwrap();

            // First subscribe
            let msg = json!({
                "method": "subscribe_graph_execution",
                "data": {
                    "graph_exec_id": "test_exec_123"
                }
            });
            ws_stream
                .send(Message::Text(msg.to_string()))
                .await
                .unwrap();
            ws_stream.next().await; // Consume response
            let msg = json!({
                "method": "unsubscribe",
                "data": {
                    "channel": "test_user|graph_exec#test_exec_123"
                }
            });

            ws_stream
                .send(Message::Text(msg.to_string()))
                .await
                .unwrap();

            // Wait for response
            if let Some(Ok(Message::Text(response))) = ws_stream.next().await {
                let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
                assert_eq!(resp["success"], true);
            }

            server_handle.abort();
        });
    });

    group.finish();
}

// Benchmark message throughput
fn benchmark_message_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("message_throughput");
    group.measurement_time(Duration::from_secs(30));

    for msg_count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*msg_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(msg_count),
            msg_count,
            |b, &msg_count| {
                b.to_async(&rt).iter_with_large_drop(|| async {
                    let (server_url, server_handle) = create_test_server(false).await;
                    let url = format!("{server_url}/ws");
                    let (mut ws_stream, _) = connect_async(&url).await.unwrap();
                    // Send multiple heartbeat messages
                    for _ in 0..msg_count {
                        let msg = json!({
                            "method": "heartbeat",
                            "data": "ping"
                        });
                        ws_stream
                            .send(Message::Text(msg.to_string()))
                            .await
                            .unwrap();
                    }

                    // Receive all responses
                    for _ in 0..msg_count {
                        ws_stream.next().await;
                    }

                    server_handle.abort();
                });
            },
        );
    }

    group.finish();
}

// Benchmark concurrent connections
fn benchmark_concurrent_connections(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_connections");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);

    for num_clients in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*num_clients as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_clients),
            num_clients,
            |b, &num_clients| {
                b.to_async(&rt).iter_with_large_drop(|| async {
                    let (server_url, server_handle) = create_test_server(false).await;
                    let url = format!("{server_url}/ws");

                    // Create multiple concurrent connections
                    let mut handles = vec![];
                    for i in 0..num_clients {
                        let url = url.clone();
                        let handle = tokio::spawn(async move {
                            let (mut ws_stream, _) = connect_async(&url).await.unwrap();

                            // Subscribe to a unique channel
                            let msg = json!({
                                "method": "subscribe_graph_execution",
                                "data": {
                                    "graph_exec_id": format!("exec_{}", i)
                                }
                            });
                            ws_stream
                                .send(Message::Text(msg.to_string()))
                                .await
                                .unwrap();
                            ws_stream.next().await; // Wait for response

                            // Send a heartbeat
                            let msg = json!({
                                "method": "heartbeat",
                                "data": "ping"
                            });
                            ws_stream
                                .send(Message::Text(msg.to_string()))
                                .await
                                .unwrap();
                            ws_stream.next().await; // Wait for response

                            ws_stream
                        });
                        handles.push(handle);
                    }

                    // Wait for all connections to complete
                    for handle in handles {
                        let _ = handle.await;
                    }

                    server_handle.abort();
                });
            },
        );
    }

    group.finish();
}

// Benchmark message parsing
fn benchmark_message_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_parsing");

    // Test different message sizes
    for msg_size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Bytes(*msg_size as u64));
        group.bench_with_input(
            BenchmarkId::new("parse_json", msg_size),
            msg_size,
            |b, &msg_size| {
                let data_str = "x".repeat(msg_size);
                let json_msg = json!({
                    "method": "subscribe_graph_execution",
                    "data": {
                        "graph_exec_id": data_str
                    }
                });
                let json_str = json_msg.to_string();

                b.iter(|| {
                    let _: models::WSMessage = serde_json::from_str(&json_str).unwrap();
                });
            },
        );
    }

    group.finish();
}

// Benchmark Redis event processing
fn benchmark_redis_event_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("redis_event_processing");

    group.bench_function("parse_execution_event", |b| {
        let event = json!({
            "payload": {
                "event_type": "graph_execution_update",
                "id": "exec_123",
                "graph_id": "graph_456",
                "graph_version": 1,
                "user_id": "user_789",
                "status": "RUNNING",
                "started_at": "2024-01-01T00:00:00Z",
                "inputs": {"test": "data"},
                "outputs": {}
            }
        });
        let event_str = event.to_string();

        b.iter(|| {
            let _: models::RedisEventWrapper = serde_json::from_str(&event_str).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_connection_establishment,
    benchmark_subscriptions,
    benchmark_message_throughput,
    benchmark_concurrent_connections,
    benchmark_message_parsing,
    benchmark_redis_event_processing
);
criterion_main!(benches);
