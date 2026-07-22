import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { HeldMemoryReviewQueue } from "./HeldMemoryReviewQueue";

const HELD_URL = "http://localhost:3000/api/proxy/api/orgs/org-1/memory/held";
const APPROVE_URL =
  "http://localhost:3000/api/proxy/api/orgs/org-1/memory/held/mem-1/approve";
const REJECT_URL =
  "http://localhost:3000/api/proxy/api/orgs/org-1/memory/held/mem-1/reject";

const TEAM_ITEM = {
  id: "mem-1",
  tier: "team",
  team_id: "team-a",
  team_name: "Growth",
  name: "Prefers async standups",
  fact: "The team prefers async standups over meetings",
  created_at: "2026-07-01T00:00:00Z",
  source_kind: "conversation",
  provenance: "from agent run #42",
};

const ORG_ITEM = {
  id: "mem-2",
  tier: "org",
  team_id: null,
  team_name: null,
  name: null,
  fact: "The company fiscal year starts in April",
  created_at: "2026-07-02T00:00:00Z",
  source_kind: "conversation",
  provenance: null,
};

describe("HeldMemoryReviewQueue", () => {
  it("renders a row per held memory with tier, team and provenance", async () => {
    server.use(
      http.get(HELD_URL, () =>
        HttpResponse.json(
          { org_id: "org-1", items: [TEAM_ITEM, ORG_ITEM] },
          { status: 200 },
        ),
      ),
    );

    render(<HeldMemoryReviewQueue orgId="org-1" />);

    await waitFor(() =>
      expect(screen.getAllByTestId("org-memory-review-row")).toHaveLength(2),
    );
    expect(screen.getByText("Growth")).toBeDefined();
    expect(
      screen.getByText("The team prefers async standups over meetings"),
    ).toBeDefined();
    expect(screen.getByText("from agent run #42")).toBeDefined();
    // Org-home held memory (null team) is labelled "Organization".
    expect(screen.getByText("Organization")).toBeDefined();
  });

  it("shows the empty state when nothing is awaiting review", async () => {
    server.use(
      http.get(HELD_URL, () =>
        HttpResponse.json({ org_id: "org-1", items: [] }, { status: 200 }),
      ),
    );

    render(<HeldMemoryReviewQueue orgId="org-1" />);

    await waitFor(() =>
      expect(screen.getByTestId("org-memory-review-empty")).toBeDefined(),
    );
    expect(screen.getByText("Nothing awaiting review.")).toBeDefined();
  });

  it("approves a held memory and refetches the queue", async () => {
    let listCalls = 0;
    let approveCalled = false;
    server.use(
      http.get(HELD_URL, () => {
        listCalls += 1;
        // First load shows the item; after approval the refetch returns empty.
        return HttpResponse.json(
          { org_id: "org-1", items: listCalls === 1 ? [TEAM_ITEM] : [] },
          { status: 200 },
        );
      }),
      http.post(APPROVE_URL, () => {
        approveCalled = true;
        return HttpResponse.json(
          { id: "mem-1", action: "approve", applied: true, tier: "team" },
          { status: 200 },
        );
      }),
    );

    render(<HeldMemoryReviewQueue orgId="org-1" />);

    const approveButton = await screen.findByRole("button", {
      name: "Approve",
    });
    await userEvent.click(approveButton);

    await waitFor(() => expect(approveCalled).toBe(true));
    await waitFor(() =>
      expect(screen.getByTestId("org-memory-review-empty")).toBeDefined(),
    );
    expect(listCalls).toBeGreaterThanOrEqual(2);
  });

  it("rejects a held memory and refetches the queue", async () => {
    let listCalls = 0;
    let rejectCalled = false;
    server.use(
      http.get(HELD_URL, () => {
        listCalls += 1;
        return HttpResponse.json(
          { org_id: "org-1", items: listCalls === 1 ? [TEAM_ITEM] : [] },
          { status: 200 },
        );
      }),
      http.post(REJECT_URL, () => {
        rejectCalled = true;
        return HttpResponse.json(
          { id: "mem-1", action: "reject", applied: true, tier: "team" },
          { status: 200 },
        );
      }),
    );

    render(<HeldMemoryReviewQueue orgId="org-1" />);

    const rejectButton = await screen.findByRole("button", { name: "Reject" });
    await userEvent.click(rejectButton);

    await waitFor(() => expect(rejectCalled).toBe(true));
    await waitFor(() =>
      expect(screen.getByTestId("org-memory-review-empty")).toBeDefined(),
    );
  });

  it("disables every action while a decision is in flight", async () => {
    server.use(
      http.get(HELD_URL, () =>
        HttpResponse.json({ org_id: "org-1", items: [TEAM_ITEM] }),
      ),
      http.post(APPROVE_URL, async () => {
        await delay(150);
        return HttpResponse.json({
          id: "mem-1",
          action: "approve",
          applied: true,
          tier: "team",
        });
      }),
    );

    render(<HeldMemoryReviewQueue orgId="org-1" />);

    const approveButton = await screen.findByRole("button", {
      name: "Approve",
    });
    const rejectButton = screen.getByRole("button", { name: "Reject" });
    await userEvent.click(approveButton);

    // Both actions lock while the approve mutation is pending.
    await waitFor(() =>
      expect(rejectButton.hasAttribute("disabled")).toBe(true),
    );
    expect(approveButton.hasAttribute("disabled")).toBe(true);
  });
});
