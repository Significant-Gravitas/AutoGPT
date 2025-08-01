use axum::{
    routing::get,
    Router,
    response::Response,
    http::{header, StatusCode},
    body::Body,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use clap::Parser;

use crate::config::Config;
use crate::connection_manager::ConnectionManager;
use crate::handlers::ws_handler;

async fn stats_handler(
    axum::Extension(state): axum::Extension<AppState>,
) -> Result<axum::response::Json<stats::StatsSnapshot>, StatusCode> {
    let snapshot = state.stats.snapshot().await;
    Ok(axum::response::Json(snapshot))
}

async fn prometheus_handler(
    axum::Extension(state): axum::Extension<AppState>,
) -> Result<Response, StatusCode> {
    let snapshot = state.stats.snapshot().await;
    let prometheus_text = state.stats.to_prometheus_format(&snapshot);
    
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(Body::from(prometheus_text))
        .unwrap())
}

mod config;
mod connection_manager;
mod handlers;
mod models;
mod stats;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Path to a TOML configuration file
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config: Option<std::path::PathBuf>,
}

#[derive(Clone)]
pub struct AppState {
    mgr: Arc<ConnectionManager>,
    config: Arc<Config>,
    stats: Arc<stats::Stats>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ws_api=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting WebSocket API server");
    
    let cli = Cli::parse();
    let config = Arc::new(Config::load(cli.config.as_deref()));
    debug!("Configuration loaded - host: {}, port: {}, auth: {}", config.host, config.port, config.enable_auth);
    
    let redis_client = match redis::Client::open(config.redis_url.clone()) {
        Ok(client) => {
            debug!("Redis client created successfully");
            client
        }
        Err(e) => {
            error!("Failed to create Redis client: {}. Please check REDIS_URL environment variable", e);
            std::process::exit(1);
        }
    };
    
    let stats = Arc::new(stats::Stats::default());
    let mgr = Arc::new(ConnectionManager::new(redis_client, config.execution_event_bus_name.clone(), stats.clone()));

    let mgr_clone = mgr.clone();
    tokio::spawn(async move {
        debug!("Starting event broadcaster task");
        mgr_clone.run_broadcaster().await;
    });

    let state = AppState {
        mgr,
        config: config.clone(),
        stats,
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/stats", get(stats_handler))
        .route("/metrics", get(prometheus_handler))
        .layer(axum::Extension(state));

    let cors = if config.backend_cors_allow_origins.is_empty() {
        // If no specific origins configured, allow any origin but without credentials
        CorsLayer::new()
            .allow_methods(Any)
            .allow_headers(Any)
            .allow_origin(Any)
    } else {
        // If specific origins configured, allow credentials
        CorsLayer::new()
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::PUT,
                axum::http::Method::DELETE,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers(vec![
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
            .allow_credentials(true)
            .allow_origin(
                config
                    .backend_cors_allow_origins
                    .iter()
                    .filter_map(|o| o.parse::<axum::http::HeaderValue>().ok())
                    .collect::<Vec<_>>(),
            )
    };

    let app = app.layer(cors);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => {
            info!("WebSocket server listening on: {}", addr);
            listener
        }
        Err(e) => {
            error!("Failed to bind to {}: {}. Please check if the port is already in use", addr, e);
            std::process::exit(1);
        }
    };
    
    info!("WebSocket API server ready to accept connections");
    
    if let Err(e) = axum::serve(listener, app.into_make_service()).await {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}
