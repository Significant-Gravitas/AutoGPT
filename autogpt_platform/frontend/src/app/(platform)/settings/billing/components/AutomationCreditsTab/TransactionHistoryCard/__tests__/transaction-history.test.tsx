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
import { executionID, libraryID, run, topUp } from "./fixtures";

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
    ).toBe(`/library/agents/${libraryID}`);
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
    ).toBe(
      `/library/agents/${libraryID}?activeTab=runs&activeItem=${executionID}`,
    );

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
});
