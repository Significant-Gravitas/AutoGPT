use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct WSMessage {
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Deserialize)]
pub struct Claims {
    pub sub: String,
}

// Event models moved from events.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum ExecutionEvent {
    #[serde(rename = "graph_execution_update")]
    GraphExecutionUpdate(GraphExecutionEvent),
    #[serde(rename = "node_execution_update")]
    NodeExecutionUpdate(NodeExecutionEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExecutionEvent {
    pub id: String,
    pub graph_id: String,
    pub graph_version: u32,
    pub user_id: String,
    pub status: ExecutionStatus,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
    pub preset_id: Option<String>,
    pub stats: Option<ExecutionStats>,

    // Keep these as JSON since they vary by graph
    pub inputs: Value,
    pub outputs: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeExecutionEvent {
    pub node_exec_id: String,
    pub node_id: String,
    pub graph_exec_id: String,
    pub graph_id: String,
    pub graph_version: u32,
    pub user_id: String,
    pub block_id: String,
    pub status: ExecutionStatus,
    pub add_time: String,
    pub queue_time: Option<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,

    // Keep these as JSON since they vary by node type
    pub input_data: Value,
    pub output_data: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub cost: f64,
    pub duration: f64,
    pub duration_cpu_only: f64,
    pub error: Option<String>,
    pub node_error_count: u32,
    pub node_exec_count: u32,
    pub node_exec_time: f64,
    pub node_exec_time_cpu_only: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Incomplete,
    Terminated,
}

// Wrapper for the Redis event that includes the payload
#[derive(Debug, Deserialize)]
pub struct RedisEventWrapper {
    pub payload: ExecutionEvent,
}

impl RedisEventWrapper {
    pub fn parse(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}
