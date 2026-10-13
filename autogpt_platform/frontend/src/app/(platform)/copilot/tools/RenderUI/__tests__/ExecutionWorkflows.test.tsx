import { fireEvent, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { showResult } from "./renderConnected";

const graph = "10000000-0000-4000-8000-000000000001";
const execution = "20000000-0000-4000-8000-000000000002";
const path = `*/api/graphs/${graph}/executions/${execution}`;
const source = `root = Workspace("Run review", "", [ExecutionView("Image run", "${graph}", "${execution}")])`;
function details(status: string) {
  return {
    id: execution,
    graph_id: graph,
    status,
    outputs: { result: ["A supplied run output"] },
    credential_inputs: { private_value: "never-render-this" },
  };
}
describe("real execution status inline", () => {
  it("renders backend outputs and waits for confirmed termination after stopping", async () => {
    let status = "RUNNING";
    let stops = 0;
    server.use(
      http.get(path, () => HttpResponse.json(details(status))),
      http.post(`${path}/stop`, () => {
        stops++;
        return HttpResponse.json({});
      }),
    );
    await showResult(vi.fn(), false, source);
    await screen.findByText("Status: running");
    expect(screen.getByText("A supplied run output")).toBeDefined();
    expect(screen.queryByText("never-render-this")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Stop execution" }));
    await screen.findByText(/waiting for the backend to confirm/);
    expect(stops).toBe(1);
    expect(
      (
        screen.getByRole("button", {
          name: "Stop execution",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.getByText("Status: running")).toBeDefined();
    status = "TERMINATED";
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Refresh status",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole("button", { name: "Refresh status" }));
    await screen.findByText("Status: terminated");
    expect(screen.queryByRole("button", { name: "Stop execution" })).toBeNull();
  });
  it("keeps a failed stop retryable and requests a failed-run retry through chat", async () => {
    let status = "RUNNING";
    const send = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue(undefined);
    server.use(
      http.get(path, () => HttpResponse.json(details(status))),
      http.post(`${path}/stop`, () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
      ),
    );
    await showResult(send, false, source);
    fireEvent.click(
      await screen.findByRole("button", { name: "Stop execution" }),
    );
    await screen.findByText(/stop request could not be confirmed/);
    expect(screen.getByText("Status: running")).toBeDefined();
    status = "FAILED";
    fireEvent.click(screen.getByRole("button", { name: "Refresh status" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Review and retry in chat" }),
    );
    await screen.findByText(/retry request could not be sent/);
    fireEvent.click(
      screen.getByRole("button", { name: "Review and retry in chat" }),
    );
    await waitFor(() => expect(send).toHaveBeenCalledTimes(2));
    expect(send.mock.calls[1][0]).toContain(execution);
    expect(send.mock.calls[1][0]).toContain("approvals");
  });
  it("does not show unauthorized status or query private executions from a share", async () => {
    let calls = 0;
    server.use(
      http.get(path, () => {
        calls++;
        return HttpResponse.json({ detail: "Forbidden" }, { status: 403 });
      }),
    );
    const view = await showResult(vi.fn(), false, source);
    await screen.findByText(/Execution details are unavailable/);
    expect(screen.queryByRole("button", { name: "Stop execution" })).toBeNull();
    const privateCalls = calls;
    view.unmount();
    await showResult(vi.fn(), true, source);
    await screen.findByText(/available in the original conversation/);
    expect(calls).toBe(privateCalls);
  });
});
