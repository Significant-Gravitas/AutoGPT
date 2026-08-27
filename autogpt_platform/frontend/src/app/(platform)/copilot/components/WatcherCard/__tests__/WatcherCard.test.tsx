import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { WatcherCard } from "../WatcherCard";
import { getWatcherMetadata } from "../helpers";

describe("WatcherCard", () => {
  it("renders semantic activity without raw diagnostic fields", () => {
    const metadata = getWatcherMetadata({
      kind: "copilot_watcher",
      title: "Lead Research needs attention",
      description: "Workflow run failed",
      action_label: "Open run",
      action_href:
        "/library/agents/library-1?activeTab=runs&activeItem=execution-1",
      status: "failed",
      execution_id: "5ff68d32-f4c4-4c6c-9f30-71a24528a100",
      graph_id: "graph-internal",
      raw_json: '{"tool_call_id":"secret"}',
    });

    expect(metadata).not.toBeNull();
    render(<WatcherCard metadata={metadata!} />);

    expect(screen.getByText("Lead Research needs attention")).toBeDefined();
    expect(screen.getByText("Workflow run failed")).toBeDefined();
    expect(
      screen.getByRole<HTMLAnchorElement>("link", { name: "Open run" }).href,
    ).toContain(
      "/library/agents/library-1?activeTab=runs&activeItem=execution-1",
    );
    expect(screen.queryByText(/5ff68d32/)).toBeNull();
    expect(screen.queryByText(/tool_call_id/)).toBeNull();
  });

  it("rejects an unsafe action link", () => {
    const metadata = getWatcherMetadata({
      kind: "copilot_watcher",
      action_href: "javascript:alert(1)",
    });

    expect(metadata?.actionHref).toBe("/home");
  });
});
