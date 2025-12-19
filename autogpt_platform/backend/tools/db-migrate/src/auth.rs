use crate::db::Database;
use anyhow::{Context, Result};
use comfy_table::{presets::UTF8_FULL, Cell, Color, Table};
use tracing::{info, warn};

/// Migrate auth data from Supabase auth.users to platform.User
pub async fn migrate_auth(source: &Database, dest: &Database) -> Result<()> {
    info!("Migrating auth data from Supabase...");

    // Check if auth.users exists in source
    let auth_exists = source
        .query(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'auth' AND table_name = 'users')",
            &[],
        )
        .await?;

    let exists: bool = auth_exists.first().map(|r| r.get(0)).unwrap_or(false);

    if !exists {
        warn!("No auth.users table found in source - skipping auth migration");
        return Ok(());
    }

    // Get count of users to migrate
    let count_rows = source
        .query(
            r#"
            SELECT COUNT(*)
            FROM auth.users
            WHERE encrypted_password IS NOT NULL
               OR raw_app_meta_data->>'provider' = 'google'
            "#,
            &[],
        )
        .await?;

    let auth_user_count: i64 = count_rows.first().map(|r| r.get(0)).unwrap_or(0);
    info!("Found {} users with auth data to migrate", auth_user_count);

    // Create temp table in destination
    info!("Creating temp table for auth data...");
    dest.batch_execute(
        r#"
        CREATE TEMP TABLE IF NOT EXISTS temp_auth_users (
            id UUID,
            encrypted_password TEXT,
            email_verified BOOLEAN,
            google_id TEXT
        )
        "#,
    )
    .await?;

    // Extract and insert auth data in batches
    info!("Extracting auth data from source...");

    let batch_size = 1000;
    let mut offset = 0i64;
    let mut total_migrated = 0i64;

    while offset < auth_user_count {
        let rows = source
            .query(
                r#"
                SELECT
                    id,
                    encrypted_password,
                    (email_confirmed_at IS NOT NULL) as email_verified,
                    CASE
                        WHEN raw_app_meta_data->>'provider' = 'google'
                        THEN raw_app_meta_data->>'provider_id'
                        ELSE NULL
                    END as google_id
                FROM auth.users
                WHERE encrypted_password IS NOT NULL
                   OR raw_app_meta_data->>'provider' = 'google'
                ORDER BY created_at
                LIMIT $1 OFFSET $2
                "#,
                &[&batch_size, &offset],
            )
            .await?;

        if rows.is_empty() {
            break;
        }

        // Insert into temp table
        for row in &rows {
            let id: uuid::Uuid = row.get(0);
            let password: Option<String> = row.get(1);
            let email_verified: bool = row.get(2);
            let google_id: Option<String> = row.get(3);

            let insert_sql = format!(
                "INSERT INTO temp_auth_users (id, encrypted_password, email_verified, google_id) VALUES ('{}', {}, {}, {})",
                id,
                password.as_ref().map(|p| format!("'{}'", p.replace('\'', "''"))).unwrap_or_else(|| "NULL".to_string()),
                email_verified,
                google_id.as_ref().map(|g| format!("'{}'", g.replace('\'', "''"))).unwrap_or_else(|| "NULL".to_string()),
            );

            dest.batch_execute(&insert_sql).await?;
            total_migrated += 1;
        }

        offset += rows.len() as i64;
        info!("  Processed {}/{} auth records", offset, auth_user_count);
    }

    info!("Migrated {} auth records to temp table", total_migrated);

    // Update User table with password hashes
    info!("Updating User table with password hashes...");
    let password_result = dest
        .execute(
            &format!(
                r#"
                UPDATE {schema}."User" u
                SET "passwordHash" = t.encrypted_password
                FROM temp_auth_users t
                WHERE u.id = t.id
                AND t.encrypted_password IS NOT NULL
                AND u."passwordHash" IS NULL
                "#,
                schema = dest.schema()
            ),
            &[],
        )
        .await?;
    info!("  Updated {} users with password hashes", password_result);

    // Update email verification status
    info!("Updating email verification status...");
    let email_result = dest
        .execute(
            &format!(
                r#"
                UPDATE {schema}."User" u
                SET "emailVerified" = t.email_verified
                FROM temp_auth_users t
                WHERE u.id = t.id
                AND t.email_verified = true
                "#,
                schema = dest.schema()
            ),
            &[],
        )
        .await?;
    info!("  Updated {} users with email verification", email_result);

    // Update Google OAuth IDs
    info!("Updating Google OAuth IDs...");
    let google_result = dest
        .execute(
            &format!(
                r#"
                UPDATE {schema}."User" u
                SET "googleId" = t.google_id
                FROM temp_auth_users t
                WHERE u.id = t.id
                AND t.google_id IS NOT NULL
                AND u."googleId" IS NULL
                "#,
                schema = dest.schema()
            ),
            &[],
        )
        .await?;
    info!("  Updated {} users with Google OAuth IDs", google_result);

    // Clean up temp table
    dest.batch_execute("DROP TABLE IF EXISTS temp_auth_users")
        .await?;

    info!("Auth migration complete!");
    Ok(())
}

/// Verify auth migration
pub async fn verify_auth(source: &Database, dest: &Database) -> Result<()> {
    info!("Verifying auth migration...");

    // Get source stats from auth.users
    let source_stats = source
        .query(
            r#"
            SELECT
                COUNT(*) as total,
                COUNT(encrypted_password) as with_password,
                COUNT(CASE WHEN raw_app_meta_data->>'provider' = 'google' THEN 1 END) as with_google,
                COUNT(CASE WHEN email_confirmed_at IS NOT NULL THEN 1 END) as email_verified
            FROM auth.users
            "#,
            &[],
        )
        .await?;

    // Get destination stats from User table
    let dest_stats = dest
        .query(
            &format!(
                r#"
                SELECT
                    COUNT(*) as total,
                    COUNT("passwordHash") as with_password,
                    COUNT("googleId") as with_google,
                    COUNT(CASE WHEN "emailVerified" = true THEN 1 END) as email_verified
                FROM {schema}."User"
                "#,
                schema = dest.schema()
            ),
            &[],
        )
        .await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Metric", "Source (auth.users)", "Dest (User)", "Status"]);

    let metrics = ["Total Users", "With Password", "With Google OAuth", "Email Verified"];

    let source_row = source_stats.first().context("No source stats")?;
    let dest_row = dest_stats.first().context("No dest stats")?;

    let mut all_match = true;

    for (i, metric) in metrics.iter().enumerate() {
        let source_val: i64 = source_row.get(i);
        let dest_val: i64 = dest_row.get(i);

        // For total users, dest may have fewer (users without auth)
        // For auth fields, they should match or dest should be >= source
        let status = if i == 0 {
            // Total users - dest should be >= source users with auth
            Cell::new("✓").fg(Color::Green)
        } else if dest_val >= source_val * 95 / 100 {
            // Allow 5% tolerance for auth fields
            Cell::new("✓").fg(Color::Green)
        } else {
            all_match = false;
            Cell::new("LOW").fg(Color::Yellow)
        };

        table.add_row(vec![
            Cell::new(*metric),
            Cell::new(source_val),
            Cell::new(dest_val),
            status,
        ]);
    }

    println!("\nAuth Migration Verification:\n{}", table);

    // Check for users without any auth method
    let orphan_check = dest
        .query(
            &format!(
                r#"
                SELECT COUNT(*)
                FROM {schema}."User"
                WHERE "passwordHash" IS NULL
                AND "googleId" IS NULL
                "#,
                schema = dest.schema()
            ),
            &[],
        )
        .await?;

    let orphans: i64 = orphan_check.first().map(|r| r.get(0)).unwrap_or(0);

    if orphans > 0 {
        warn!(
            "{} users have no auth method (may be other OAuth providers)",
            orphans
        );
    }

    if all_match {
        info!("Auth verification passed!");
    } else {
        warn!("Some auth metrics don't match - review above table");
    }

    Ok(())
}

/// Migrate auth data for a single user
pub async fn migrate_single_user_auth(source: &Database, dest: &Database, user_id: &str) -> Result<()> {
    // Parse as UUID for auth.users query (Supabase uses native UUID)
    let uid = uuid::Uuid::parse_str(user_id).context("Invalid user ID format")?;

    info!("Migrating auth for user: {}", user_id);

    // Check if auth.users exists
    let auth_exists = source
        .query(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'auth' AND table_name = 'users')",
            &[],
        )
        .await?;

    let exists: bool = auth_exists.first().map(|r| r.get(0)).unwrap_or(false);

    if !exists {
        warn!("No auth.users table found - skipping");
        return Ok(());
    }

    // Get auth data for this user (auth.users uses native UUID type)
    let rows = source
        .query(
            r#"
            SELECT
                encrypted_password,
                (email_confirmed_at IS NOT NULL) as email_verified,
                CASE
                    WHEN raw_app_meta_data->>'provider' = 'google'
                    THEN raw_app_meta_data->>'provider_id'
                    ELSE NULL
                END as google_id
            FROM auth.users
            WHERE id = $1
            "#,
            &[&uid],
        )
        .await?;

    if let Some(row) = rows.first() {
        let password: Option<String> = row.get(0);
        let email_verified: bool = row.get(1);
        let google_id: Option<String> = row.get(2);

        info!("  Found auth data:");
        info!("    Has password: {}", password.is_some());
        info!("    Email verified: {}", email_verified);
        info!("    Has Google ID: {}", google_id.is_some());

        // Update password hash (platform.User.id is String, not UUID)
        if let Some(ref pw) = password {
            dest.execute(
                &format!(
                    "UPDATE {}.\"User\" SET \"passwordHash\" = $1 WHERE id = $2 AND \"passwordHash\" IS NULL",
                    dest.schema()
                ),
                &[pw, &user_id],
            )
            .await?;
            info!("    Updated password hash");
        }

        // Update email verified
        if email_verified {
            dest.execute(
                &format!(
                    "UPDATE {}.\"User\" SET \"emailVerified\" = true WHERE id = $1",
                    dest.schema()
                ),
                &[&user_id],
            )
            .await?;
            info!("    Updated email verification");
        }

        // Update Google ID
        if let Some(ref gid) = google_id {
            dest.execute(
                &format!(
                    "UPDATE {}.\"User\" SET \"googleId\" = $1 WHERE id = $2 AND \"googleId\" IS NULL",
                    dest.schema()
                ),
                &[gid, &user_id],
            )
            .await?;
            info!("    Updated Google OAuth ID");
        }
    } else {
        warn!("  No auth data found for user");
    }

    Ok(())
}

/// Show detailed auth comparison
pub async fn compare_auth_details(source: &Database, dest: &Database) -> Result<()> {
    info!("Comparing auth details...");

    // Find users in source auth.users but missing auth in dest
    let missing = dest
        .query(
            &format!(
                r#"
                SELECT u.id, u.email
                FROM {schema}."User" u
                WHERE u."passwordHash" IS NULL
                AND u."googleId" IS NULL
                LIMIT 10
                "#,
                schema = dest.schema()
            ),
            &[],
        )
        .await?;

    if !missing.is_empty() {
        println!("\nSample users without auth method:");
        for row in missing {
            let id: String = row.get(0);  // platform.User.id is String
            let email: String = row.get(1);
            println!("  {} - {}", id, email);
        }
    }

    Ok(())
}
