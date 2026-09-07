import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { TransactionHistoryCard } from "../TransactionHistoryCard";
import { executionID, run, topUp } from "./fixtures";

describe("Transaction history receipts", () => {
  it("links the agent separately from its task and labels run status in the receipt", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({ transactions: [run], next_cursor: null }),
      ),
    );
    render(<TransactionHistoryCard />);

    expect(
      (
        await screen.findByRole("link", { name: "Morning briefing" })
      ).getAttribute("href"),
    ).toBe(run.agent_url);
    expect(screen.queryByRole("link", { name: "View task" })).toBeNull();
    const details = screen.getByRole("button", {
      name: /details for Morning briefing/i,
    });
    expect(details.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(details);

    const receipt = screen.getByRole("region", {
      name: "Morning briefing credit receipt",
    });
    expect(
      within(receipt).getByRole("heading", { name: "Charges for this run" })
        .textContent,
    ).toMatch(/^Charges for this run$/);
    const statusLabel = within(receipt).getByText("Run status");
    expect(statusLabel.closest("div")?.textContent).toContain("Running");
    expect(within(receipt).getByText("Net change so far")).toBeDefined();
    expect(
      within(receipt).getByText("+$0.03").classList.contains("text-green-700"),
    ).toBe(true);
    expect(
      within(receipt)
        .getByRole("link", { name: /View task/ })
        .getAttribute("href"),
    ).toBe(run.execution_url);

    fireEvent.click(
      within(receipt).getByRole("button", { name: /Show charge entries/ }),
    );
    expect(
      within(receipt).getByRole("table", { name: "Recorded charge entries" }),
    ).toBeDefined();
    fireEvent.click(within(receipt).getByRole("button", { name: "Reference" }));
    expect(within(receipt).getByText(executionID)).toBeDefined();
  });

  it("preserves loaded receipts when loading more fails and retries the same cursor", async () => {
    let attempts = 0;
    server.use(
      http.get("*/api/credits/transactions", ({ request }) => {
        if (!new URL(request.url).searchParams.has("cursor"))
          return HttpResponse.json({
            transactions: [run],
            next_cursor: "older-page",
          });
        attempts += 1;
        if (attempts === 1)
          return HttpResponse.json({ detail: "Unavailable" }, { status: 500 });
        expect(new URL(request.url).searchParams.get("cursor")).toBe(
          "older-page",
        );
        return HttpResponse.json({ transactions: [topUp], next_cursor: null });
      }),
    );
    render(<TransactionHistoryCard />);
    await screen.findByRole("link", { name: "Morning briefing" });
    fireEvent.click(
      screen.getByRole("button", { name: /details for Morning briefing/i }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Load more/ }));
    await screen.findByText(/couldn.t load older transactions/i);
    expect(
      screen.getByRole("region", { name: "Morning briefing credit receipt" }),
    ).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: /Retry loading more/ }));
    expect(await screen.findByText("Credits added")).toBeDefined();
    expect(
      screen.getByText("+$20.00").classList.contains("text-green-700"),
    ).toBe(true);
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull(),
    );
  });

  it("keeps unavailable history without creating a dead library or task link", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            {
              ...run,
              agent_name: null,
              library_agent_id: null,
              agent_url: null,
              execution_url: null,
              execution_available: false,
              execution_status: null,
              execution_started_at: null,
              execution_graph_version: null,
            },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    await screen.findByText("Agent unavailable");
    expect(screen.queryByRole("link")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: /details for Agent unavailable/i }),
    );
    expect(screen.getByText("Agent and run unavailable")).toBeDefined();
    expect(screen.queryByRole("link", { name: /View task/ })).toBeNull();
  });

  it("keeps the agent link when its run is unavailable without creating a dead task link", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            { ...run, execution_available: false, execution_url: null },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    expect(
      (
        await screen.findByRole("link", { name: "Morning briefing" })
      ).getAttribute("href"),
    ).toBe(run.agent_url);
    fireEvent.click(
      screen.getByRole("button", { name: /details for Morning briefing/i }),
    );
    expect(screen.queryByRole("link", { name: /View task/ })).toBeNull();
    expect(screen.getByText("Run unavailable")).toBeDefined();
  });

  it("keeps complete totals when the charge-entry list is limited", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            { ...run, charges_total_count: 130, charges_truncated: true },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    await screen.findByText("Morning briefing");
    fireEvent.click(
      screen.getByRole("button", { name: /details for Morning briefing/i }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /Show charge entries/ }),
    );
    expect(screen.getByText(/Showing 3 of 130 charge entries/)).toBeDefined();
    expect(
      screen.getByText(/The total includes every charge and adjustment/),
    ).toBeDefined();
  });

  it("uses each authorized source URL for parent and related runs", async () => {
    const parentURL =
      "/library/agents/parent-library?activeTab=runs&activeItem=parent-run&organizationId=__personal__&teamId=__org_home__";
    const relatedURL =
      "/library/agents/related-library?activeTab=runs&activeItem=related-run&organizationId=other-org&teamId=other-team";
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            {
              ...run,
              parent_execution_id: "parent-run",
              parent_library_agent_id: "parent-library",
              parent_agent_name: "Parent agent",
              parent_execution_url: parentURL,
              related_executions: [
                {
                  execution_id: "related-run",
                  library_agent_id: "related-library",
                  agent_name: "Related agent",
                  execution_available: true,
                  execution_url: relatedURL,
                },
              ],
            },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    fireEvent.click(
      await screen.findByRole("button", {
        name: /details for Morning briefing/i,
      }),
    );
    expect(
      screen.getByRole("link", { name: "Parent agent" }).getAttribute("href"),
    ).toBe(parentURL);
    expect(
      screen.getByRole("link", { name: "Related agent" }).getAttribute("href"),
    ).toBe(relatedURL);
  });

  it("does not reconstruct unavailable destinations from retained reference IDs", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            {
              ...run,
              agent_url: null,
              execution_url: null,
              conversation_id: "reference-chat",
              conversation_url: null,
              parent_execution_id: "parent-run",
              parent_library_agent_id: "parent-library",
              parent_agent_name: "Parent unavailable",
              parent_execution_url: null,
              related_executions: [
                {
                  execution_id: "related-run",
                  library_agent_id: "related-library",
                  execution_available: true,
                  agent_name: "Related unavailable",
                  execution_url: null,
                },
              ],
            },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    fireEvent.click(
      await screen.findByRole("button", {
        name: /details for Morning briefing/i,
      }),
    );
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("Parent unavailable")).toBeDefined();
    expect(screen.getByText("Related unavailable")).toBeDefined();
    expect(screen.getByText("Run unavailable")).toBeDefined();
  });
});
