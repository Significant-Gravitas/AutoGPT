import { useSyncExternalStore } from "react";

export type CoPilotCostEstimate = {
  resolved_path: "baseline" | "sdk";
  resolved_model: string;
  estimated_llm_calls: number;
  estimated_cost_usd: number;
  confirmation_threshold_usd: number;
  rationale: string;
  requires_confirmation: boolean;
  approval_token: string | null;
};

type PendingCost = {
  estimate: CoPilotCostEstimate;
  resolve: (token: string | null) => void;
};

let pending: PendingCost | null = null;
const listeners = new Set<() => void>();

export class CostConfirmationCancelled extends Error {
  constructor() {
    super("Cost confirmation cancelled");
    this.name = "CostConfirmationCancelled";
  }
}

export function isCostConfirmationCancelled(error: unknown): boolean {
  return (
    error instanceof CostConfirmationCancelled ||
    (error instanceof Error &&
      error.message.includes("Cost confirmation cancelled"))
  );
}

function emitPendingCost() {
  for (const listener of listeners) listener();
}

export function subscribePendingCost(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getPendingCostEstimate() {
  return pending?.estimate ?? null;
}

export function usePendingCostEstimate() {
  return useSyncExternalStore(
    subscribePendingCost,
    getPendingCostEstimate,
    () => null,
  );
}

export function confirmPendingCost() {
  finishPendingCost(pending?.estimate.approval_token ?? null);
}

export function cancelPendingCost() {
  finishPendingCost(null);
}

function finishPendingCost(token: string | null) {
  const current = pending;
  pending = null;
  emitPendingCost();
  current?.resolve(token);
}

function waitForCostApproval(estimate: CoPilotCostEstimate) {
  if (pending) finishPendingCost(null);
  return new Promise<string | null>((resolve) => {
    pending = { estimate, resolve };
    emitPendingCost();
  });
}

export function fetchWithCostApproval(inner: typeof fetch): typeof fetch {
  return async (input, init) => {
    const response = await inner(input, init);
    const estimate = await confirmationEstimate(response);
    if (!estimate?.approval_token) return response;
    const token = await waitForCostApproval(estimate);
    if (!token) throw new CostConfirmationCancelled();
    const raw = typeof init?.body === "string" ? init.body : null;
    if (raw === null) throw new CostConfirmationCancelled();
    const body = JSON.stringify({
      ...JSON.parse(raw),
      approval_token: token,
    });
    return inner(input, { ...init, body });
  };
}

async function confirmationEstimate(response: Response) {
  if (response.status !== 403) return null;
  try {
    const body = (await response.clone().json()) as CoPilotCostEstimate;
    if (!body?.requires_confirmation || !body.approval_token) return null;
    return body;
  } catch {
    return null;
  }
}
