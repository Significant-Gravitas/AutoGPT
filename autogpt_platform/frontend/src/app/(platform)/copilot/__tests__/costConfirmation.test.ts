import { afterEach, describe, expect, it, vi } from "vitest";

import {
  cancelPendingCost,
  confirmPendingCost,
  fetchWithCostApproval,
  getPendingCostEstimate,
} from "../costConfirmation";

const estimate = {
  resolved_path: "sdk",
  resolved_model: "claude-opus-4-1",
  estimated_llm_calls: 2,
  estimated_cost_usd: 1.5,
  confirmation_threshold_usd: 1,
  rationale: "sdk path",
  requires_confirmation: true,
  approval_token: "signed-token",
};

afterEach(() => {
  cancelPendingCost();
  vi.unstubAllGlobals();
});

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("fetchWithCostApproval", () => {
  it("leaves a turn that does not need confirmation untouched", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, "stream"));
    const wrapped = fetchWithCostApproval(fetchImpl);

    const response = await wrapped("http://backend.test/stream", {
      method: "POST",
      body: JSON.stringify({ message: "hello" }),
    });

    expect(response.status).toBe(200);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(getPendingCostEstimate()).toBeNull();
  });

  it("retries the same request with the approval token after continue", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(403, estimate))
      .mockResolvedValueOnce(jsonResponse(200, "stream"));
    const wrapped = fetchWithCostApproval(fetchImpl);
    const pending = wrapped("http://backend.test/stream", {
      method: "POST",
      body: JSON.stringify({ message: "hello" }),
    });
    await vi.waitFor(() => {
      expect(getPendingCostEstimate()?.approval_token).toBe("signed-token");
    });
    confirmPendingCost();

    await expect(pending).resolves.toEqual(
      expect.objectContaining({ status: 200 }),
    );
    const retry = fetchImpl.mock.calls[1]?.[1];
    expect(JSON.parse(String(retry?.body))).toEqual({
      message: "hello",
      approval_token: "signed-token",
    });
  });

  it("does not retry when the estimate is cancelled", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(403, estimate));
    const wrapped = fetchWithCostApproval(fetchImpl);
    const pending = wrapped("http://backend.test/stream", {
      method: "POST",
      body: JSON.stringify({ message: "hello" }),
    });
    await vi.waitFor(() => {
      expect(getPendingCostEstimate()).not.toBeNull();
    });
    cancelPendingCost();

    await expect(pending).rejects.toThrow(/Cost confirmation cancelled/);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});
