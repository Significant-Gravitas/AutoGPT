use crate::db::Database;
use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Table, Cell, Color};
use tracing::{info, warn, error};

/// Show table sizes in the database
pub async fn show_table_sizes(db: &Database) -> Result<()> {
    let tables = db.get_tables().await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Table", "Rows", "Size"]);

    let mut total_bytes: i64 = 0;
    let mut total_rows: i64 = 0;

    for t in &tables {
        let count = db.get_row_count(t).await?;
        let (bytes, size) = db.get_table_size(t).await?;

        total_bytes += bytes;
        total_rows += count;

        table.add_row(vec![t.clone(), count.to_string(), size]);
    }

    println!("\n{}", table);
    println!(
        "\nTotal: {} rows, {} bytes ({:.2} GB)",
        total_rows,
        total_bytes,
        total_bytes as f64 / 1_073_741_824.0
    );

    Ok(())
}

/// Verify row counts match between source and destination
pub async fn verify_row_counts(source: &Database, dest: &Database) -> Result<()> {
    info!("Verifying row counts...");

    let tables = source.get_tables().await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Table", "Source", "Dest", "Status"]);

    let mut all_match = true;
    let mut total_source: i64 = 0;
    let mut total_dest: i64 = 0;

    for t in &tables {
        let source_count = source.get_row_count(t).await?;
        let dest_count = dest.get_row_count(t).await.unwrap_or(0);

        total_source += source_count;
        total_dest += dest_count;

        let status = if source_count == dest_count {
            Cell::new("✓").fg(Color::Green)
        } else if dest_count == 0 {
            all_match = false;
            Cell::new("MISSING").fg(Color::Yellow)
        } else {
            all_match = false;
            Cell::new("MISMATCH").fg(Color::Red)
        };

        table.add_row(vec![
            Cell::new(t),
            Cell::new(source_count),
            Cell::new(dest_count),
            status,
        ]);
    }

    println!("\n{}", table);
    println!("\nTotal: Source={}, Dest={}", total_source, total_dest);

    if all_match {
        info!("All row counts match!");
    } else {
        warn!("Some tables have mismatched row counts");
    }

    Ok(())
}

/// Verify functions and triggers exist in destination
pub async fn verify_functions(source: &Database, dest: &Database) -> Result<()> {
    info!("Verifying functions...");

    let source_funcs = source.get_functions().await?;
    let dest_funcs = dest.get_functions().await?;

    let dest_func_names: std::collections::HashSet<_> =
        dest_funcs.iter().map(|(n, _)| n.clone()).collect();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Function", "Status"]);

    let mut all_present = true;

    for (name, _def) in &source_funcs {
        let status = if dest_func_names.contains(name) {
            Cell::new("✓").fg(Color::Green)
        } else {
            all_present = false;
            Cell::new("MISSING").fg(Color::Red)
        };

        table.add_row(vec![Cell::new(name), status]);
    }

    println!("\nFunctions:\n{}", table);

    // Verify triggers
    info!("Verifying triggers...");

    let source_triggers = source.get_triggers().await?;
    let dest_triggers = dest.get_triggers().await?;

    let dest_trigger_names: std::collections::HashSet<_> =
        dest_triggers.iter().map(|(n, _, _)| n.clone()).collect();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Trigger", "Table", "Status"]);

    for (name, tbl, _def) in &source_triggers {
        let status = if dest_trigger_names.contains(name) {
            Cell::new("✓").fg(Color::Green)
        } else {
            all_present = false;
            Cell::new("MISSING").fg(Color::Red)
        };

        table.add_row(vec![Cell::new(name), Cell::new(tbl), status]);
    }

    println!("\nTriggers:\n{}", table);

    // Verify materialized views
    info!("Verifying materialized views...");

    let source_views = source.get_materialized_views().await?;
    let dest_views = dest.get_materialized_views().await?;

    let dest_view_names: std::collections::HashSet<_> = dest_views.into_iter().collect();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Materialized View", "Status"]);

    for name in &source_views {
        let status = if dest_view_names.contains(name) {
            Cell::new("✓").fg(Color::Green)
        } else {
            all_present = false;
            Cell::new("MISSING").fg(Color::Red)
        };

        table.add_row(vec![Cell::new(name), status]);
    }

    println!("\nMaterialized Views:\n{}", table);

    if all_present {
        info!("All functions, triggers, and views present!");
    } else {
        error!("Some database objects are missing in destination");
    }

    Ok(())
}

/// Verify data integrity with checksums
pub async fn verify_checksums(source: &Database, dest: &Database, table: &str) -> Result<bool> {
    info!("Computing checksums for {}...", table);

    // Get checksum of all data
    let checksum_sql = format!(
        r#"
        SELECT md5(string_agg(t::text, ''))
        FROM (SELECT * FROM {}."{}" ORDER BY 1) t
        "#,
        source.schema(),
        table
    );

    let source_rows = source.query(&checksum_sql, &[]).await?;
    let dest_rows = dest.query(&checksum_sql, &[]).await?;

    let source_checksum: Option<String> = source_rows.first().and_then(|r| r.get(0));
    let dest_checksum: Option<String> = dest_rows.first().and_then(|r| r.get(0));

    match (source_checksum, dest_checksum) {
        (Some(s), Some(d)) if s == d => {
            info!("  {} checksum match: {}", table, s);
            Ok(true)
        }
        (Some(s), Some(d)) => {
            error!("  {} checksum MISMATCH: {} vs {}", table, s, d);
            Ok(false)
        }
        _ => {
            warn!("  {} could not compute checksum", table);
            Ok(false)
        }
    }
}
