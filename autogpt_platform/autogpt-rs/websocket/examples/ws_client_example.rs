use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "ws://localhost:8001/ws";

    println!("Connecting to {url}");
    let (mut ws_stream, _) = connect_async(url).await?;
    println!("Connected!");

    // Subscribe to a graph execution
    let subscribe_msg = json!({
        "method": "subscribe_graph_execution",
        "data": {
            "graph_exec_id": "test_exec_123"
        }
    });

    println!("Sending subscription request...");
    ws_stream
        .send(Message::Text(subscribe_msg.to_string()))
        .await?;

    // Wait for response
    if let Some(msg) = ws_stream.next().await {
        if let Message::Text(text) = msg? {
            println!("Received: {text}");
        }
    }

    // Send heartbeat
    let heartbeat_msg = json!({
        "method": "heartbeat",
        "data": "ping"
    });

    println!("Sending heartbeat...");
    ws_stream
        .send(Message::Text(heartbeat_msg.to_string()))
        .await?;

    // Wait for pong
    if let Some(msg) = ws_stream.next().await {
        if let Message::Text(text) = msg? {
            println!("Received: {text}");
        }
    }

    // Unsubscribe
    let unsubscribe_msg = json!({
        "method": "unsubscribe",
        "data": {
            "channel": "default|graph_exec#test_exec_123"
        }
    });

    println!("Sending unsubscribe request...");
    ws_stream
        .send(Message::Text(unsubscribe_msg.to_string()))
        .await?;

    // Wait for response
    if let Some(msg) = ws_stream.next().await {
        if let Message::Text(text) = msg? {
            println!("Received: {text}");
        }
    }

    println!("Closing connection...");
    ws_stream.close(None).await?;

    Ok(())
}
