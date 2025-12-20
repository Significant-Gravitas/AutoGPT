use anyhow::{Context, Result};
use tokio_postgres::{Client, NoTls, Row};
use url::Url;

pub struct Database {
    client: Client,
    schema: String,
    host: String,
}

impl Database {
    pub async fn connect(url: &str, schema: &str) -> Result<Self> {
        // Parse URL to extract host for display
        let parsed = Url::parse(url).context("Invalid database URL")?;
        let host = parsed.host_str().unwrap_or("unknown").to_string();

        // Remove schema parameter from URL for tokio-postgres
        let base_url = url.split('?').next().unwrap_or(url);

        let (client, connection) = tokio_postgres::connect(base_url, NoTls)
            .await
            .context("Failed to connect to database")?;

        // Spawn connection handler
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("Database connection error: {}", e);
            }
        });

        // Set search path to schema
        client
            .execute(&format!("SET search_path TO {}", schema), &[])
            .await
            .context("Failed to set search_path")?;

        Ok(Self {
            client,
            schema: schema.to_string(),
            host,
        })
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub async fn query(&self, sql: &str, params: &[&(dyn tokio_postgres::types::ToSql + Sync)]) -> Result<Vec<Row>> {
        self.client
            .query(sql, params)
            .await
            .context("Query failed")
    }

    pub async fn execute(&self, sql: &str, params: &[&(dyn tokio_postgres::types::ToSql + Sync)]) -> Result<u64> {
        self.client
            .execute(sql, params)
            .await
            .context("Execute failed")
    }

    pub async fn batch_execute(&self, sql: &str) -> Result<()> {
        self.client
            .batch_execute(sql)
            .await
            .with_context(|| format!("Batch execute failed:\n{}", sql.chars().take(500).collect::<String>()))
    }

    /// Get all table names in the schema
    pub async fn get_tables(&self) -> Result<Vec<String>> {
        let rows = self
            .query(
                "SELECT tablename FROM pg_tables WHERE schemaname = $1 ORDER BY tablename",
                &[&self.schema],
            )
            .await?;

        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    /// Get row count for a table
    pub async fn get_row_count(&self, table: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {}.\"{}\"", self.schema, table);
        let rows = self.query(&sql, &[]).await?;
        Ok(rows[0].get(0))
    }

    /// Get table size
    pub async fn get_table_size(&self, table: &str) -> Result<(i64, String)> {
        let sql = format!(
            "SELECT pg_total_relation_size('{}.\"{}\"'), pg_size_pretty(pg_total_relation_size('{}.\"{}\"'))",
            self.schema, table, self.schema, table
        );
        let rows = self.query(&sql, &[]).await?;
        Ok((rows[0].get(0), rows[0].get(1)))
    }

    /// Check if table exists
    pub async fn table_exists(&self, table: &str) -> Result<bool> {
        let rows = self
            .query(
                "SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = $1 AND tablename = $2)",
                &[&self.schema, &table],
            )
            .await?;
        Ok(rows[0].get(0))
    }

    /// Get functions in schema
    pub async fn get_functions(&self) -> Result<Vec<(String, String)>> {
        let rows = self
            .query(
                r#"
                SELECT p.proname, pg_get_functiondef(p.oid)
                FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = $1
                ORDER BY p.proname
                "#,
                &[&self.schema],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| (r.get::<_, String>(0), r.get::<_, String>(1)))
            .collect())
    }

    /// Get triggers in schema
    pub async fn get_triggers(&self) -> Result<Vec<(String, String, String)>> {
        let rows = self
            .query(
                r#"
                SELECT
                    t.tgname,
                    c.relname as table_name,
                    pg_get_triggerdef(t.oid)
                FROM pg_trigger t
                JOIN pg_class c ON t.tgrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = $1
                AND NOT t.tgisinternal
                ORDER BY c.relname, t.tgname
                "#,
                &[&self.schema],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get::<_, String>(0),
                    r.get::<_, String>(1),
                    r.get::<_, String>(2),
                )
            })
            .collect())
    }

    /// Get materialized views
    pub async fn get_materialized_views(&self) -> Result<Vec<String>> {
        let rows = self
            .query(
                r#"
                SELECT matviewname
                FROM pg_matviews
                WHERE schemaname = $1
                ORDER BY matviewname
                "#,
                &[&self.schema],
            )
            .await?;

        Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
    }

    /// Copy data from table using COPY protocol (for streaming)
    pub async fn copy_out(&self, table: &str) -> Result<Vec<u8>> {
        let sql = format!(
            "COPY {}.\"{}\" TO STDOUT WITH (FORMAT binary)",
            self.schema, table
        );

        let stream = self
            .client
            .copy_out(&sql)
            .await
            .context("COPY OUT failed")?;

        use futures::StreamExt;
        use tokio_postgres::binary_copy::BinaryCopyOutStream;

        let mut data = Vec::new();
        let mut stream = std::pin::pin!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading COPY stream")?;
            data.extend_from_slice(&chunk);
        }

        Ok(data)
    }
}
