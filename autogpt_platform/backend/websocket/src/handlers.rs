use axum::{
    extract::{Query, WebSocketUpgrade},
    http::HeaderMap,
    response::IntoResponse,
    Extension,
};
use axum::extract::ws::{CloseFrame, Message, WebSocket};
use jsonwebtoken::{decode, Validation, DecodingKey};
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{error, info, warn, debug};

use crate::connection_manager::ConnectionManager;
use crate::models::{Claims, WSMessage};
use crate::AppState;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    query: Query<HashMap<String, String>>,
    _headers: HeaderMap,
    Extension(state): Extension<AppState>,
) -> impl IntoResponse {
    let token = query.0.get("token").cloned();
    let mut user_id = state.config.default_user_id.clone();
    let mut auth_error_code: Option<u16> = None;

    if state.config.enable_auth {
        match token {
            Some(token_str) => {
                debug!("Authenticating WebSocket connection");
            let mut validation = Validation::new(state.config.jwt_algorithm);
            validation.set_audience(&["authenticated"]);

            let key = DecodingKey::from_secret(state.config.jwt_secret.as_bytes());

                match decode::<Claims>(&token_str, &key, &validation) {
                    Ok(token_data) => {
                        user_id = token_data.claims.sub.clone();
                        debug!("WebSocket authenticated for user: {}", user_id);
                    }
                    Err(e) => {
                        warn!("JWT validation failed: {}", e);
                        auth_error_code = Some(4003);
                    }
                }
            }
            None => {
                warn!("Missing authentication token in WebSocket connection");
                auth_error_code = Some(4001);
            }
        }
    } else {
        debug!("WebSocket connection without auth (auth disabled)");
    }

    if let Some(code) = auth_error_code {
        error!("WebSocket authentication failed with code: {}", code);
        state.mgr.stats.connections_failed_auth.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        state.mgr.stats.connections_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return ws
            .on_upgrade(move |mut socket: WebSocket| async move {
                let close_frame = Some(CloseFrame {
                    code,
                    reason: "Authentication failed".into(),
                });
                let _ = socket.send(Message::Close(close_frame)).await;
                let _ = socket.close().await;
            })
            .into_response();
    }

    debug!("WebSocket connection established for user: {}", user_id);
    ws.on_upgrade(move |socket| {
        handle_socket(socket, user_id, state.mgr.clone(), state.config.max_message_size_limit)
    })
}

async fn update_subscription_stats(mgr: &ConnectionManager, channel: &str, add: bool) {
    if add {
        mgr.stats.subscriptions_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        mgr.stats.subscriptions_active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let mut channel_stats = mgr.stats.channels_active.write().await;
        let count = channel_stats.entry(channel.to_string()).or_insert(0);
        *count += 1;
    } else {
        mgr.stats.unsubscriptions_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        mgr.stats.subscriptions_active.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        
        let mut channel_stats = mgr.stats.channels_active.write().await;
        if let Some(count) = channel_stats.get_mut(channel) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                channel_stats.remove(channel);
            }
        }
    }
}

pub async fn handle_socket(
    mut socket: WebSocket,
    user_id: String,
    mgr: std::sync::Arc<ConnectionManager>,
    max_size: usize,
) {
    let client_id = mgr.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (tx, mut rx) = mpsc::channel::<String>(10);
    info!("New WebSocket client {} for user: {}", client_id, user_id);
    
    // Update connection stats
    mgr.stats.connections_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    mgr.stats.connections_active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    // Update active users
    {
        let mut active_users = mgr.stats.active_users.write().await;
        let count = active_users.entry(user_id.clone()).or_insert(0);
        *count += 1;
    }

    {
        let mut clients = mgr.clients.write().await;
        clients.insert(client_id, (user_id.clone(), tx));
    }

    {
        let mut client_channels = mgr.client_channels.write().await;
        client_channels.insert(client_id, std::collections::HashSet::new());
    }

    loop {
        tokio::select! {
            msg = rx.recv() => {
                if let Some(msg) = msg {
                    if socket.send(Message::Text(msg)).await.is_err() {
                        break;
                    }
                } else {
                    break;
                }
            }
            incoming = socket.recv() => {
                let msg = match incoming {
                    Some(Ok(msg)) => msg,
                    _ => break,
                };
                match msg {
                    Message::Text(text) => {
                        if text.len() > max_size {
                            warn!("Message from client {} exceeds size limit: {} > {}", client_id, text.len(), max_size);
                            let err_resp = serde_json::to_string(&WSMessage {
                                method: "error".to_string(),
                                success: Some(false),
                                error: Some("Message exceeds size limit".to_string()),
                                ..Default::default()
                            }).unwrap();
                            if socket.send(Message::Text(err_resp)).await.is_err() {
                                break;
                            }
                            continue;
                        }

                        mgr.stats.messages_received_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        
                        let ws_msg: WSMessage = match serde_json::from_str(&text) {
                            Ok(m) => m,
                            Err(e) => {
                                warn!("Invalid message format from client {}: {}", client_id, e);
                                mgr.stats.errors_json_parse.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                mgr.stats.errors_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let err_resp = serde_json::to_string(&WSMessage {
                                    method: "error".to_string(),
                                    success: Some(false),
                                    error: Some("Invalid message format. Review the schema and retry".to_string()),
                                    ..Default::default()
                                }).unwrap();
                                if socket.send(Message::Text(err_resp)).await.is_err() {
                                    break;
                                }
                                continue;
                            }
                        };
                        
                        debug!("Received {} message from client {}", ws_msg.method, client_id);
                        
                        match ws_msg.method.as_str() {
                            "subscribe_graph_execution" => {
                                let graph_exec_id = match &ws_msg.data {
                                    Some(Value::Object(map)) => map.get("graph_exec_id").and_then(|v| v.as_str()),
                                    _ => None,
                                };
                                let Some(graph_exec_id) = graph_exec_id else {
                                    warn!("Missing graph_exec_id in subscribe_graph_execution from client {}", client_id);
                                    let err_resp = json!({"method": "error", "success": false, "error": "Missing graph_exec_id"});
                                    if socket.send(Message::Text(err_resp.to_string())).await.is_err() {
                                        break;
                                    }
                                    continue;
                                };
                                let channel = format!("{}|graph_exec#{}", user_id, graph_exec_id);
                                debug!("Client {} subscribing to channel: {}", client_id, channel);

                                {
                                    let mut subs = mgr.subscribers.write().await;
                                    subs.entry(channel.clone()).or_insert(std::collections::HashSet::new()).insert(client_id);
                                }
                                {
                                    let mut chs = mgr.client_channels.write().await;
                                    if let Some(set) = chs.get_mut(&client_id) {
                                        set.insert(channel.clone());
                                    }
                                }
                                
                                // Update subscription stats
                                update_subscription_stats(&mgr, &channel, true).await;

                                let resp = WSMessage {
                                    method: "subscribe_graph_execution".to_string(),
                                    success: Some(true),
                                    channel: Some(channel),
                                    ..Default::default()
                                };
                                if socket.send(Message::Text(serde_json::to_string(&resp).unwrap())).await.is_err() {
                                    break;
                                }
                            }
                            "subscribe_graph_executions" => {
                                let graph_id = match &ws_msg.data {
                                    Some(Value::Object(map)) => map.get("graph_id").and_then(|v| v.as_str()),
                                    _ => None,
                                };
                                let Some(graph_id) = graph_id else {
                                    let err_resp = json!({"method": "error", "success": false, "error": "Missing graph_id"});
                                    if socket.send(Message::Text(err_resp.to_string())).await.is_err() {
                                        break;
                                    }
                                    continue;
                                };
                                let channel = format!("{}|graph#{}|executions", user_id, graph_id);

                                {
                                    let mut subs = mgr.subscribers.write().await;
                                    subs.entry(channel.clone()).or_insert(std::collections::HashSet::new()).insert(client_id);
                                }
                                {
                                    let mut chs = mgr.client_channels.write().await;
                                    if let Some(set) = chs.get_mut(&client_id) {
                                        set.insert(channel.clone());
                                    }
                                }
                                
                                // Update subscription stats
                                update_subscription_stats(&mgr, &channel, true).await;

                                let resp = WSMessage {
                                    method: "subscribe_graph_executions".to_string(),
                                    success: Some(true),
                                    channel: Some(channel),
                                    ..Default::default()
                                };
                                if socket.send(Message::Text(serde_json::to_string(&resp).unwrap())).await.is_err() {
                                    break;
                                }
                            }
                            "unsubscribe" => {
                                let channel = match &ws_msg.data {
                                    Some(Value::String(s)) => Some(s.as_str()),
                                    Some(Value::Object(map)) => map.get("channel").and_then(|v| v.as_str()),
                                    _ => None,
                                };
                                let Some(channel) = channel else {
                                    let err_resp = json!({"method": "error", "success": false, "error": "Missing channel"});
                                    if socket.send(Message::Text(err_resp.to_string())).await.is_err() {
                                        break;
                                    }
                                    continue;
                                };
                                let channel = channel.to_string();

                                if !channel.starts_with(&format!("{}|", user_id)) {
                                    let err_resp = json!({"method": "error", "success": false, "error": "Unauthorized channel"});
                                    if socket.send(Message::Text(err_resp.to_string())).await.is_err() {
                                        break;
                                    }
                                    continue;
                                }

                                {
                                    let mut subs = mgr.subscribers.write().await;
                                    if let Some(set) = subs.get_mut(&channel) {
                                        set.remove(&client_id);
                                        if set.is_empty() {
                                            subs.remove(&channel);
                                        }
                                    }
                                }
                                {
                                    let mut chs = mgr.client_channels.write().await;
                                    if let Some(set) = chs.get_mut(&client_id) {
                                        set.remove(&channel);
                                    }
                                }
                                
                                // Update subscription stats
                                update_subscription_stats(&mgr, &channel, false).await;

                                let resp = WSMessage {
                                    method: "unsubscribe".to_string(),
                                    success: Some(true),
                                    channel: Some(channel),
                                    ..Default::default()
                                };
                                if socket.send(Message::Text(serde_json::to_string(&resp).unwrap())).await.is_err() {
                                    break;
                                }
                            }
                            "heartbeat" => {
                                if ws_msg.data == Some(Value::String("ping".to_string())) {
                                    let resp = WSMessage {
                                        method: "heartbeat".to_string(),
                                        data: Some(Value::String("pong".to_string())),
                                        success: Some(true),
                                        ..Default::default()
                                    };
                                    if socket.send(Message::Text(serde_json::to_string(&resp).unwrap())).await.is_err() {
                                        break;
                                    }
                                } else {
                                    let err_resp = json!({"method": "error", "success": false, "error": "Invalid heartbeat"});
                                    if socket.send(Message::Text(err_resp.to_string())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            _ => {
                                warn!("Unknown method '{}' from client {}", ws_msg.method, client_id);
                                let err_resp = json!({"method": "error", "success": false, "error": "Unknown method"});
                                if socket.send(Message::Text(err_resp.to_string())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Message::Close(_) => break,
                    Message::Ping(_) => {
                        if socket.send(Message::Pong(vec![])).await.is_err() {
                            break;
                        }
                    }
                    Message::Pong(_) => {}
                    _ => {}
                }
            }
            else => break,
        }
    }

    // Cleanup
    debug!("WebSocket client {} disconnected, cleaning up", client_id);
    
    // Update connection stats
    mgr.stats.connections_active.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    
    // Update active users
    {
        let mut active_users = mgr.stats.active_users.write().await;
        if let Some(count) = active_users.get_mut(&user_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                active_users.remove(&user_id);
            }
        }
    }
    
    let channels = {
        let mut client_channels = mgr.client_channels.write().await;
        client_channels.remove(&client_id).unwrap_or_default()
    };

    {
        let mut subs = mgr.subscribers.write().await;
        for channel in &channels {
            if let Some(set) = subs.get_mut(channel) {
                set.remove(&client_id);
                if set.is_empty() {
                    subs.remove(channel);
                }
            }
        }
    }
    
    // Update subscription stats for all channels the client was subscribed to
    for channel in &channels {
        update_subscription_stats(&mgr, channel, false).await;
    }

    {
        let mut clients = mgr.clients.write().await;
        clients.remove(&client_id);
    }
    
    debug!("Cleanup completed for client {}", client_id);
}
