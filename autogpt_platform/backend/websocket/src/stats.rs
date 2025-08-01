use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

#[derive(Default)]
pub struct Stats {
    // Connection metrics
    pub connections_total: AtomicU64,
    pub connections_active: AtomicU64,
    pub connections_failed_auth: AtomicU64,

    // Message metrics
    pub messages_received_total: AtomicU64,
    pub messages_sent_total: AtomicU64,
    pub messages_failed_total: AtomicU64,

    // Subscription metrics
    pub subscriptions_total: AtomicU64,
    pub subscriptions_active: AtomicU64,
    pub unsubscriptions_total: AtomicU64,

    // Event metrics by type
    pub events_received_total: AtomicU64,
    pub graph_execution_events: AtomicU64,
    pub node_execution_events: AtomicU64,

    // Redis metrics
    pub redis_messages_received: AtomicU64,
    pub redis_messages_ignored: AtomicU64,

    // Channel metrics
    pub channels_active: RwLock<HashMap<String, usize>>, // channel -> subscriber count

    // User metrics
    pub active_users: RwLock<HashMap<String, usize>>, // user_id -> connection count

    // Error metrics
    pub errors_total: AtomicU64,
    pub errors_json_parse: AtomicU64,
    pub errors_message_size: AtomicU64,
}

#[derive(Serialize, Deserialize)]
pub struct StatsSnapshot {
    // Connection metrics
    pub connections_total: u64,
    pub connections_active: u64,
    pub connections_failed_auth: u64,

    // Message metrics
    pub messages_received_total: u64,
    pub messages_sent_total: u64,
    pub messages_failed_total: u64,

    // Subscription metrics
    pub subscriptions_total: u64,
    pub subscriptions_active: u64,
    pub unsubscriptions_total: u64,

    // Event metrics
    pub events_received_total: u64,
    pub graph_execution_events: u64,
    pub node_execution_events: u64,

    // Redis metrics
    pub redis_messages_received: u64,
    pub redis_messages_ignored: u64,

    // Channel metrics
    pub channels_active_count: usize,
    pub total_subscribers: usize,

    // User metrics
    pub active_users_count: usize,

    // Error metrics
    pub errors_total: u64,
    pub errors_json_parse: u64,
    pub errors_message_size: u64,
}

impl Stats {
    pub async fn snapshot(&self) -> StatsSnapshot {
        // Take read locks for HashMap data - it's ok if this is slightly stale
        let channels = self.channels_active.read().await;
        let total_subscribers: usize = channels.values().sum();
        let channels_active_count = channels.len();
        drop(channels); // Release lock early

        let users = self.active_users.read().await;
        let active_users_count = users.len();
        drop(users); // Release lock early

        StatsSnapshot {
            connections_total: self.connections_total.load(Ordering::Relaxed),
            connections_active: self.connections_active.load(Ordering::Relaxed),
            connections_failed_auth: self.connections_failed_auth.load(Ordering::Relaxed),

            messages_received_total: self.messages_received_total.load(Ordering::Relaxed),
            messages_sent_total: self.messages_sent_total.load(Ordering::Relaxed),
            messages_failed_total: self.messages_failed_total.load(Ordering::Relaxed),

            subscriptions_total: self.subscriptions_total.load(Ordering::Relaxed),
            subscriptions_active: self.subscriptions_active.load(Ordering::Relaxed),
            unsubscriptions_total: self.unsubscriptions_total.load(Ordering::Relaxed),

            events_received_total: self.events_received_total.load(Ordering::Relaxed),
            graph_execution_events: self.graph_execution_events.load(Ordering::Relaxed),
            node_execution_events: self.node_execution_events.load(Ordering::Relaxed),

            redis_messages_received: self.redis_messages_received.load(Ordering::Relaxed),
            redis_messages_ignored: self.redis_messages_ignored.load(Ordering::Relaxed),

            channels_active_count,
            total_subscribers,
            active_users_count,

            errors_total: self.errors_total.load(Ordering::Relaxed),
            errors_json_parse: self.errors_json_parse.load(Ordering::Relaxed),
            errors_message_size: self.errors_message_size.load(Ordering::Relaxed),
        }
    }

    pub fn to_prometheus_format(&self, snapshot: &StatsSnapshot) -> String {
        let mut output = String::new();

        // Connection metrics
        output.push_str("# HELP ws_connections_total Total number of WebSocket connections\n");
        output.push_str("# TYPE ws_connections_total counter\n");
        output.push_str(&format!(
            "ws_connections_total {}\n\n",
            snapshot.connections_total
        ));

        output.push_str(
            "# HELP ws_connections_active Current number of active WebSocket connections\n",
        );
        output.push_str("# TYPE ws_connections_active gauge\n");
        output.push_str(&format!(
            "ws_connections_active {}\n\n",
            snapshot.connections_active
        ));

        output
            .push_str("# HELP ws_connections_failed_auth Total number of failed authentications\n");
        output.push_str("# TYPE ws_connections_failed_auth counter\n");
        output.push_str(&format!(
            "ws_connections_failed_auth {}\n\n",
            snapshot.connections_failed_auth
        ));

        // Message metrics
        output.push_str(
            "# HELP ws_messages_received_total Total number of messages received from clients\n",
        );
        output.push_str("# TYPE ws_messages_received_total counter\n");
        output.push_str(&format!(
            "ws_messages_received_total {}\n\n",
            snapshot.messages_received_total
        ));

        output.push_str("# HELP ws_messages_sent_total Total number of messages sent to clients\n");
        output.push_str("# TYPE ws_messages_sent_total counter\n");
        output.push_str(&format!(
            "ws_messages_sent_total {}\n\n",
            snapshot.messages_sent_total
        ));

        // Subscription metrics
        output.push_str("# HELP ws_subscriptions_active Current number of active subscriptions\n");
        output.push_str("# TYPE ws_subscriptions_active gauge\n");
        output.push_str(&format!(
            "ws_subscriptions_active {}\n\n",
            snapshot.subscriptions_active
        ));

        // Event metrics
        output.push_str(
            "# HELP ws_events_received_total Total number of events received from Redis\n",
        );
        output.push_str("# TYPE ws_events_received_total counter\n");
        output.push_str(&format!(
            "ws_events_received_total {}\n\n",
            snapshot.events_received_total
        ));

        output.push_str(
            "# HELP ws_graph_execution_events_total Total number of graph execution events\n",
        );
        output.push_str("# TYPE ws_graph_execution_events_total counter\n");
        output.push_str(&format!(
            "ws_graph_execution_events_total {}\n\n",
            snapshot.graph_execution_events
        ));

        output.push_str(
            "# HELP ws_node_execution_events_total Total number of node execution events\n",
        );
        output.push_str("# TYPE ws_node_execution_events_total counter\n");
        output.push_str(&format!(
            "ws_node_execution_events_total {}\n\n",
            snapshot.node_execution_events
        ));

        // Channel metrics
        output.push_str("# HELP ws_channels_active Number of active channels\n");
        output.push_str("# TYPE ws_channels_active gauge\n");
        output.push_str(&format!(
            "ws_channels_active {}\n\n",
            snapshot.channels_active_count
        ));

        output.push_str(
            "# HELP ws_total_subscribers Total number of subscribers across all channels\n",
        );
        output.push_str("# TYPE ws_total_subscribers gauge\n");
        output.push_str(&format!(
            "ws_total_subscribers {}\n\n",
            snapshot.total_subscribers
        ));

        // User metrics
        output.push_str("# HELP ws_active_users Number of unique users with active connections\n");
        output.push_str("# TYPE ws_active_users gauge\n");
        output.push_str(&format!(
            "ws_active_users {}\n\n",
            snapshot.active_users_count
        ));

        // Error metrics
        output.push_str("# HELP ws_errors_total Total number of errors\n");
        output.push_str("# TYPE ws_errors_total counter\n");
        output.push_str(&format!("ws_errors_total {}\n", snapshot.errors_total));

        output
    }
}
