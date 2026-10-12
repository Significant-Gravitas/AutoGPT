export const executionID = "e4288112-28f2-432e-b5f0-937d22a7bc21";
export const libraryID = "3b8c1a22-5c27-48aa-ae42-184cff161bb2";

export const run = {
  id: `execution:${executionID}`,
  transaction_key: `execution:${executionID}`,
  transaction_type: "USAGE",
  transaction_time: "2026-09-05T08:14:09Z",
  amount: -12,
  description: "Agent run",
  activity_type: "agent_run",
  agent_name: "Morning briefing",
  library_agent_id: libraryID,
  usage_execution_id: executionID,
  execution_available: true,
  execution_started_at: "2026-09-05T08:13:52Z",
  execution_status: "RUNNING",
  execution_graph_version: 3,
  usage_charge_amount: -14,
  usage_fee_amount: -1,
  usage_adjustment_amount: 3,
  related_executions: [],
  charges_total_count: 3,
  charges_truncated: false,
  charges: [
    {
      id: "charge-1",
      amount: -14,
      posted_at: "2026-09-05T08:13:54Z",
      charge_type: "usage",
    },
    {
      id: "charge-2",
      amount: -1,
      posted_at: "2026-09-05T08:13:58Z",
      charge_type: "execution_fee",
    },
    {
      id: "charge-3",
      amount: 3,
      posted_at: "2026-09-05T08:14:09Z",
      charge_type: "adjustment",
    },
  ],
};

export const topUp = {
  id: "transaction:topup-1",
  transaction_key: "topup-1",
  transaction_type: "TOP_UP",
  transaction_time: "2026-09-04T08:30:12Z",
  description: "Credits added",
  activity_type: "other",
  amount: 2000,
  charges: [],
  related_executions: [],
};
