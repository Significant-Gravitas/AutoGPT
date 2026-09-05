_HISTORY_QUERY = """
WITH wallet AS MATERIALIZED (
    SELECT "transactionKey", "createdAt", amount, type::text AS transaction_type,
           metadata
    FROM {schema_prefix}__LEDGER__
    WHERE __OWNER__ = $1 AND "isActive" = TRUE
      AND "createdAt" <= ($2::timestamptz AT TIME ZONE 'UTC')
      AND ($3::text IS NULL OR type::text = $3::text)
), classified AS MATERIALIZED (
    SELECT *,
        CASE WHEN transaction_type = 'USAGE'
                  AND NULLIF(metadata->>'graph_exec_id', '') IS NOT NULL
             THEN 'execution:' || (metadata->>'graph_exec_id')
             ELSE 'transaction:' || "transactionKey" END AS group_id,
        CASE WHEN transaction_type != 'USAGE' THEN 'transaction'
             WHEN metadata->'input' ? 'reconciled_delta' THEN 'adjustment'
             WHEN metadata->'input'->>'charge' = 'Execution Cost' THEN 'execution_fee'
             ELSE 'usage' END AS charge_type
    FROM wallet
), grouped AS (
    SELECT group_id AS id,
        CASE WHEN group_id LIKE 'execution:%' THEN group_id
             ELSE MAX("transactionKey") END AS transaction_key,
        MAX("createdAt") AS transaction_time,
        MIN("createdAt") AS usage_start_time,
        MAX(transaction_type) AS transaction_type,
        SUM(amount)::bigint AS amount,
        MAX(CASE WHEN transaction_type = 'USAGE'
                 THEN NULLIF(metadata->>'graph_id', '') END) AS usage_graph_id,
        MAX(CASE WHEN transaction_type = 'USAGE'
                 THEN NULLIF(metadata->>'graph_exec_id', '') END) AS usage_execution_id,
        COUNT(DISTINCT CASE WHEN transaction_type = 'USAGE'
                           THEN NULLIF(metadata->>'node_exec_id', '') END)::int
            AS usage_node_count,
        BOOL_OR(NULLIF(metadata->>'block_id', '') IS NOT NULL
                OR NULLIF(metadata->>'block', '') IS NOT NULL) AS usage_has_block,
        BOOL_OR(COALESCE(metadata->>'reason', '') = 'CoPilot daily rate limit reset')
            AS usage_is_daily_reset,
        COALESCE(SUM(amount) FILTER (WHERE charge_type = 'usage'), 0)::bigint
            AS usage_charge_amount,
        COALESCE(SUM(amount) FILTER (WHERE charge_type = 'execution_fee'), 0)::bigint
            AS usage_fee_amount,
        COALESCE(SUM(amount) FILTER (WHERE charge_type = 'adjustment'), 0)::bigint
            AS usage_adjustment_amount,
        COUNT(*)::int AS charges_total_count
    FROM classified
    GROUP BY group_id
), page AS (
    SELECT * FROM grouped
    WHERE ($4::timestamptz IS NULL
           OR (transaction_time, id) < (($4::timestamptz AT TIME ZONE 'UTC'), $5::text))
      AND ($6::timestamptz IS NULL
           OR transaction_time < ($6::timestamptz AT TIME ZONE 'UTC'))
    ORDER BY transaction_time DESC, id DESC
    LIMIT $7::int
), selected_entries AS (
    SELECT classified.*, ROW_NUMBER() OVER (
        PARTITION BY group_id ORDER BY "createdAt", "transactionKey"
    ) AS entry_number
    FROM classified INNER JOIN page ON page.id = classified.group_id
), details AS (
    SELECT group_id, jsonb_agg(jsonb_build_object(
        'id', "transactionKey",
        'posted_at', "createdAt",
        'amount', amount,
        'charge_type', charge_type,
        'block_name', LEFT(metadata->>'block', 200),
        'node_execution_id', metadata->>'node_exec_id'
    ) ORDER BY "createdAt", "transactionKey") AS charges
    FROM selected_entries WHERE entry_number <= 100
    GROUP BY group_id
)
SELECT page.*, COALESCE(details.charges, '[]'::jsonb) AS charges
FROM page LEFT JOIN details ON details.group_id = page.id
ORDER BY page.transaction_time DESC, page.id DESC
"""


def credit_history_query(*, organization: bool) -> str:
    ledger, owner = (
        ('"OrgCreditTransaction"', '"orgId"')
        if organization
        else ('"CreditTransaction"', '"userId"')
    )
    return _HISTORY_QUERY.replace("__LEDGER__", ledger).replace("__OWNER__", owner)
