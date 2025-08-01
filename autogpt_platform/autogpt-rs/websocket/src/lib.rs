#![deny(warnings)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::panic)]
#![deny(clippy::unimplemented)]
#![deny(clippy::todo)]


pub mod config;
pub mod connection_manager;
pub mod handlers;
pub mod models;
pub mod stats;

pub use config::Config;
pub use connection_manager::ConnectionManager;
pub use handlers::ws_handler;
pub use stats::Stats;

use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub mgr: Arc<ConnectionManager>,
    pub config: Arc<Config>,
    pub stats: Arc<Stats>,
}
