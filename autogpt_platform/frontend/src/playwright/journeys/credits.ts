import { expect, type Page } from "@playwright/test";

interface CreditTransaction {
  transaction_key: string;
  transaction_type: string;
  amount: number;
  usage_execution_id: string | null;
}

export interface CreditSnapshot {
  balance: number;
  transactionKeys: Set<string>;
}

export async function getCreditBalance(page: Page) {
  const response = await page.request.get("/api/proxy/api/credits");
  expect(response.ok(), `GET credits: ${response.status()}`).toBe(true);
  const body = (await response.json()) as { credits: number };
  return body.credits;
}

export async function getCreditTransactions(page: Page) {
  const response = await page.request.get(
    "/api/proxy/api/credits/transactions",
    { params: { transaction_count_limit: 100 } },
  );
  expect(response.ok(), `GET credit transactions: ${response.status()}`).toBe(
    true,
  );
  const body = (await response.json()) as {
    transactions: CreditTransaction[];
  };
  return body.transactions;
}

export async function snapshotCredits(page: Page): Promise<CreditSnapshot> {
  const [balance, transactions] = await Promise.all([
    getCreditBalance(page),
    getCreditTransactions(page),
  ]);
  return {
    balance,
    transactionKeys: new Set(transactions.map((t) => t.transaction_key)),
  };
}

/**
 * What changed in the ledger since `before`: the new balance, every new
 * transaction, and how much of it was charged to `executionID`.
 */
export async function creditsSince(
  page: Page,
  before: CreditSnapshot,
  executionID: string,
) {
  const [balance, transactions] = await Promise.all([
    getCreditBalance(page),
    getCreditTransactions(page),
  ]);
  const added = transactions.filter(
    (t) => !before.transactionKeys.has(t.transaction_key),
  );
  const chargedToRun = added
    .filter((t) => t.usage_execution_id === executionID)
    .reduce((sum, t) => sum - t.amount, 0);
  const netChange = added.reduce((sum, t) => sum + t.amount, 0);
  return { balance, added, chargedToRun, netChange };
}
