use redis::Client as RedisClient;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn, debug};
use futures::StreamExt;

use crate::models::{WSMessage, RedisEventWrapper, ExecutionEvent};
use crate::stats::Stats;

pub struct ConnectionManager {
    pub subscribers: RwLock<HashMap<String, HashSet<u64>>>,
    pub clients: RwLock<HashMap<u64, (String, mpsc::Sender<String>)>>,
    pub client_channels: RwLock<HashMap<u64, HashSet<String>>>,
    pub next_id: AtomicU64,
    pub redis_client: RedisClient,
    pub bus_name: String,
    pub stats: Arc<Stats>,
}

impl ConnectionManager {
    pub fn new(redis_client: RedisClient, bus_name: String, stats: Arc<Stats>) -> Self {
        Self {
            subscribers: RwLock::new(HashMap::new()),
            clients: RwLock::new(HashMap::new()),
            client_channels: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            redis_client,
            bus_name,
            stats,
        }
    }

    pub async fn run_broadcaster(self: Arc<Self>) {
        info!("Starting Redis event broadcaster");
        
        loop {
            match self.run_broadcaster_inner().await {
                Ok(_) => {
                    warn!("Event broadcaster stopped unexpectedly, restarting in 5 seconds");
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
                Err(e) => {
                    error!("Event broadcaster error: {}, restarting in 5 seconds", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    }
    
    async fn run_broadcaster_inner(self: &Arc<Self>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut pubsub = self.redis_client.get_async_pubsub().await?;
        pubsub.psubscribe("*").await?;
        debug!("Listening to all Redis events, filtering for bus: {}", self.bus_name);

        let mut pubsub_stream = pubsub.on_message();
        
        loop {
            let msg = pubsub_stream.next().await;
            match msg {
                Some(msg) => {
                    let channel: String = msg.get_channel_name().to_string();
                    debug!("Received message on Redis channel: {}", channel);
                    self.stats.redis_messages_received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    
                    let payload: String = match msg.get_payload() {
                        Ok(p) => p,
                        Err(e) => {
                            warn!("Failed to get payload from Redis message: {}", e);
                            self.stats.errors_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                    };

                    // Parse the channel format: execution_event/{user_id}/{graph_id}/{graph_exec_id}
                    let parts: Vec<&str> = channel.split('/').collect();
                    
                    // Check if this is an execution event channel
                    if parts.len() != 4 || parts[0] != &self.bus_name {
                        debug!("Ignoring non-execution event channel: {} (parts: {:?}, bus_name: {})", channel, parts, self.bus_name);
                        self.stats.redis_messages_ignored.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                    
                    let user_id = parts[1];
                    let graph_id = parts[2];
                    let graph_exec_id = parts[3];
                    
                    debug!("Received event - user: {}, graph: {}, exec: {}", user_id, graph_id, graph_exec_id);

                    // Parse the wrapped event
                    let wrapped_event = match RedisEventWrapper::parse(&payload) {
                        Ok(e) => e,
                        Err(e) => {
                            warn!("Failed to parse event JSON: {}, payload: {}", e, payload);
                            self.stats.errors_json_parse.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            self.stats.errors_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                    };

                    let event = wrapped_event.payload;
                    debug!("Event received: {:?}", event);
                    
                    let (method, event_json) = match &event {
                        ExecutionEvent::GraphExecutionUpdate(graph_event) => {
                            self.stats.graph_execution_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            self.stats.events_received_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            ("graph_execution_event", serde_json::to_value(graph_event).unwrap())
                        },
                        ExecutionEvent::NodeExecutionUpdate(node_event) => {
                            self.stats.node_execution_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            self.stats.events_received_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            ("node_execution_event", serde_json::to_value(node_event).unwrap())
                        },
                    };
                    
                    // Create the channel keys in the format expected by WebSocket clients
                    let mut channels_to_notify = Vec::new();
                    
                    // For both event types, notify the specific execution channel
                    let exec_channel = format!("{}|graph_exec#{}", user_id, graph_exec_id);
                    channels_to_notify.push(exec_channel.clone());
                    
                    // For graph execution events, also notify the graph executions channel
                    if matches!(&event, ExecutionEvent::GraphExecutionUpdate(_)) {
                        let graph_channel = format!("{}|graph#{}|executions", user_id, graph_id);
                        channels_to_notify.push(graph_channel);
                    }
                    
                    debug!("Broadcasting {} event to channels: {:?}", method, channels_to_notify);
                    
                    let subs = self.subscribers.read().await;
                    
                    for channel_key in channels_to_notify {
                        let ws_msg = WSMessage {
                            method: method.to_string(),
                            channel: Some(channel_key.clone()),
                            data: Some(event_json.clone()),
                            ..Default::default()
                        };
                        let json_msg = match serde_json::to_string(&ws_msg) {
                            Ok(j) => {
                                debug!("Sending WebSocket message: {}", j);
                                j
                            },
                            Err(e) => {
                                error!("Failed to serialize WebSocket message: {}", e);
                                continue;
                            }
                        };
                        
                        if let Some(client_ids) = subs.get(&channel_key) {
                        let clients = self.clients.read().await;
                        let client_count = client_ids.len();
                        debug!("Broadcasting to {} clients on channel: {}", client_count, channel_key);
                        
                        for &cid in client_ids {
                            if let Some((user_id, tx)) = clients.get(&cid) {
                                match tx.try_send(json_msg.clone()) {
                                    Ok(_) => {
                                        debug!("Message sent immediately to client {} (user: {})", cid, user_id);
                                    }
                                    Err(mpsc::error::TrySendError::Full(_)) => {
                                        // Channel is full, try with a small timeout
                                        let tx_clone = tx.clone();
                                        let msg_clone = json_msg.clone();
                                        tokio::spawn(async move {
                                            let _ = tokio::time::timeout(
                                                std::time::Duration::from_millis(100),
                                                tx_clone.send(msg_clone)
                                            ).await;
                                        });
                                        warn!("Channel full for client {} (user: {}), sending async", cid, user_id);
                                    }
                                    Err(mpsc::error::TrySendError::Closed(_)) => {
                                        warn!("Channel closed for client {} (user: {})", cid, user_id);
                                    }
                                }
                            }
                        }
                        } else {
                            debug!("No subscribers for channel: {}", channel_key);
                        }
                    }
                }
                None => {
                    return Err("Redis pubsub stream ended".into());
                }
            }
        }
    }
}
