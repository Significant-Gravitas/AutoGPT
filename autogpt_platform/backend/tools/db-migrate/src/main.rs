use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::{fmt, EnvFilter};

mod db;
mod migrate;
mod verify;
mod auth;

#[derive(Parser)]
#[command(name = "db-migrate")]
#[command(about = "Database migration tool for AutoGPT Platform")]
struct Cli {
    /// Source database URL (Supabase)
    #[arg(long, env = "SOURCE_URL")]
    source: String,

    /// Destination database URL (GCP Cloud SQL)
    #[arg(long, env = "DEST_URL")]
    dest: String,

    /// Schema name (default: platform)
    #[arg(long, default_value = "platform")]
    schema: String,

    /// Dry run mode - verify everything works without making changes
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run full migration (schema + data + auth + verify)
    Full {
        /// Skip large execution tables
        #[arg(long, default_value = "true")]
        skip_large_tables: bool,
    },

    /// Quick migration: User, Profile, UserOnboarding, UserBalance + auth (for testing)
    Quick,

    /// Solo run: migrate a single user's data for testing
    Solo {
        /// User ID to migrate (uses first user if not specified)
        #[arg(long)]
        user_id: Option<String>,
    },

    /// Migrate schema only
    Schema,

    /// Migrate data only (assumes schema exists)
    Data {
        /// Skip large execution tables
        #[arg(long, default_value = "true")]
        skip_large_tables: bool,

        /// Specific table to migrate
        #[arg(long)]
        table: Option<String>,
    },

    /// Migrate auth data (passwords, OAuth IDs)
    Auth,

    /// Verify both databases match
    Verify {
        /// Check triggers and functions
        #[arg(long)]
        check_functions: bool,
    },

    /// Show table sizes in source
    TableSizes,

    /// Stream large tables (execution history)
    StreamLarge {
        /// Specific table to stream
        #[arg(long)]
        table: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("db_migrate=info".parse()?))
        .init();

    let cli = Cli::parse();

    info!("Connecting to databases...");
    let source = db::Database::connect(&cli.source, &cli.schema).await?;
    let dest = db::Database::connect(&cli.dest, &cli.schema).await?;

    info!("Source: {}", source.host());
    info!("Destination: {}", dest.host());

    if cli.dry_run {
        warn!("DRY RUN MODE - No changes will be made");
    }

    match cli.command {
        Commands::Full { skip_large_tables } => {
            info!("=== Running Full Migration ===");

            // Step 1: Migrate schema
            info!("\n=== Step 1: Migrating Schema ===");
            migrate::migrate_schema(&source, &dest).await?;

            // Step 2: Migrate data
            info!("\n=== Step 2: Migrating Data ===");
            migrate::migrate_data(&source, &dest, skip_large_tables).await?;

            // Step 3: Verify data
            info!("\n=== Step 3: Verifying Data ===");
            verify::verify_row_counts(&source, &dest).await?;

            // Step 4: Migrate auth
            info!("\n=== Step 4: Migrating Auth Data ===");
            auth::migrate_auth(&source, &dest).await?;

            // Step 5: Verify auth
            info!("\n=== Step 5: Verifying Auth Migration ===");
            auth::verify_auth(&source, &dest).await?;

            // Step 6: Check functions/triggers
            info!("\n=== Step 6: Checking Functions & Triggers ===");
            verify::verify_functions(&source, &dest).await?;

            info!("\n=== Migration Complete! ===");
        }

        Commands::Quick => {
            info!("=== Quick Migration: Users, Profiles, Auth ===");

            let quick_tables = vec![
                "User",
                "Profile",
                "UserOnboarding",
                "UserBalance",
            ];

            // Step 1: Migrate schema for quick tables
            info!("\n=== Step 1: Creating Schema ===");
            migrate::migrate_schema(&source, &dest).await?;

            // Step 1.5: Verify all quick tables exist in destination
            info!("\n=== Step 1.5: Verifying Tables Exist ===");
            for table in &quick_tables {
                let exists = dest.table_exists(table).await?;
                if !exists {
                    anyhow::bail!("Table {} was not created in destination! Check schema migration errors.", table);
                }
                info!("  ✓ {} exists", table);
            }

            // Step 2: Migrate user-related tables
            info!("\n=== Step 2: Migrating User Tables ===");
            for table in &quick_tables {
                info!("Migrating {}...", table);
                migrate::migrate_table(&source, &dest, table).await?;
            }

            // Step 3: Migrate auth
            info!("\n=== Step 3: Migrating Auth Data ===");
            auth::migrate_auth(&source, &dest).await?;

            // Step 4: Verify
            info!("\n=== Step 4: Verification ===");
            for table in &quick_tables {
                let source_count = source.get_row_count(table).await?;
                let dest_count = dest.get_row_count(table).await?;
                let status = if source_count == dest_count { "✓" } else { "✗" };
                info!("  {}: {} -> {} {}", table, source_count, dest_count, status);
            }
            auth::verify_auth(&source, &dest).await?;

            info!("\n=== Quick Migration Complete! ===");
            info!("You can now test user login/signup");
        }

        Commands::Solo { user_id } => {
            info!("=== Solo Run: Single User Migration ===");

            // Get a user ID to migrate
            let uid = if let Some(id) = user_id {
                id
            } else {
                // Get first user from source (id is stored as String in Prisma)
                let rows = source
                    .query(
                        &format!("SELECT id FROM {}.\"User\" LIMIT 1", source.schema()),
                        &[],
                    )
                    .await?;
                let id: String = rows.first().context("No users found")?.get(0);
                id
            };

            info!("Migrating user: {}", uid);

            // Create schema
            info!("\n=== Step 1: Creating Schema ===");
            migrate::migrate_schema(&source, &dest).await?;

            // Migrate single user
            info!("\n=== Step 2: Migrating Single User ===");
            migrate::migrate_single_user(&source, &dest, &uid).await?;

            // Migrate auth for this user
            info!("\n=== Step 3: Migrating Auth ===");
            auth::migrate_single_user_auth(&source, &dest, &uid).await?;

            // Verify
            info!("\n=== Step 4: Verification ===");
            let dest_user = dest
                .query(
                    &format!("SELECT id, email, \"passwordHash\" IS NOT NULL as has_pw, \"googleId\" IS NOT NULL as has_google FROM {}.\"User\" WHERE id = $1", dest.schema()),
                    &[&uid],
                )
                .await?;

            if let Some(row) = dest_user.first() {
                let email: String = row.get(1);
                let has_pw: bool = row.get(2);
                let has_google: bool = row.get(3);
                info!("  Email: {}", email);
                info!("  Has password: {}", has_pw);
                info!("  Has Google OAuth: {}", has_google);
            }

            info!("\n=== Solo Run Complete! ===");
        }

        Commands::Schema => {
            migrate::migrate_schema(&source, &dest).await?;
        }

        Commands::Data { skip_large_tables, table } => {
            if let Some(table_name) = table {
                migrate::migrate_table(&source, &dest, &table_name).await?;
            } else {
                migrate::migrate_data(&source, &dest, skip_large_tables).await?;
            }
        }

        Commands::Auth => {
            auth::migrate_auth(&source, &dest).await?;
            auth::verify_auth(&source, &dest).await?;
        }

        Commands::Verify { check_functions } => {
            verify::verify_row_counts(&source, &dest).await?;
            if check_functions {
                verify::verify_functions(&source, &dest).await?;
            }
        }

        Commands::TableSizes => {
            verify::show_table_sizes(&source).await?;
        }

        Commands::StreamLarge { table } => {
            migrate::stream_large_tables(&source, &dest, table).await?;
        }
    }

    Ok(())
}
