use jsonwebtoken::Algorithm;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use toml;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub jwt_secret: String,
    pub jwt_algorithm: Algorithm,
    pub execution_event_bus_name: String,
    pub redis_url: String,
    pub default_user_id: String,
    pub max_message_size_limit: usize,
    pub backend_cors_allow_origins: Vec<String>,
}

impl Config {
    pub fn load(config_path: Option<&Path>) -> Self {
        let path = config_path.unwrap_or(Path::new("config.toml"));
        let toml_result = fs::read_to_string(path)
            .ok()
            .and_then(|s| toml::from_str::<Config>(&s).ok());

        let mut config = match toml_result {
            Some(config) => config,
            None => Config {
                host: env::var("WEBSOCKET_SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("WEBSOCKET_SERVER_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8001),
                enable_auth: env::var("ENABLE_AUTH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(true),
                jwt_secret: env::var("SUPABASE_JWT_SECRET")
                    .unwrap_or_else(|_| "dummy_secret_for_no_auth".to_string()),
                jwt_algorithm: Algorithm::HS256,
                execution_event_bus_name: env::var("EXECUTION_EVENT_BUS_NAME")
                    .unwrap_or_else(|_| "execution_event".to_string()),
                redis_url: env::var("REDIS_URL")
                    .unwrap_or_else(|_| "redis://localhost/".to_string()),
                default_user_id: "default".to_string(),
                max_message_size_limit: env::var("MAX_MESSAGE_SIZE_LIMIT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(512000),
                backend_cors_allow_origins: env::var("BACKEND_CORS_ALLOW_ORIGINS")
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            },
        };

        if let Ok(v) = env::var("WEBSOCKET_SERVER_HOST") {
            config.host = v;
        }
        if let Ok(v) = env::var("WEBSOCKET_SERVER_PORT") {
            config.port = v.parse().unwrap_or(8001);
        }
        if let Ok(v) = env::var("ENABLE_AUTH") {
            config.enable_auth = v.parse().unwrap_or(true);
        }
        if let Ok(v) = env::var("SUPABASE_JWT_SECRET") {
            config.jwt_secret = v;
        }
        if let Ok(v) = env::var("JWT_ALGORITHM") {
            config.jwt_algorithm = Algorithm::from_str(&v).unwrap_or(Algorithm::HS256);
        }
        if let Ok(v) = env::var("EXECUTION_EVENT_BUS_NAME") {
            config.execution_event_bus_name = v;
        }
        if let Ok(v) = env::var("REDIS_URL") {
            config.redis_url = v;
        }
        if let Ok(v) = env::var("DEFAULT_USER_ID") {
            config.default_user_id = v;
        }
        if let Ok(v) = env::var("MAX_MESSAGE_SIZE_LIMIT") {
            config.max_message_size_limit = v.parse().unwrap_or(512000);
        }
        if let Ok(v) = env::var("BACKEND_CORS_ALLOW_ORIGINS") {
            config.backend_cors_allow_origins = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        config
    }
}
