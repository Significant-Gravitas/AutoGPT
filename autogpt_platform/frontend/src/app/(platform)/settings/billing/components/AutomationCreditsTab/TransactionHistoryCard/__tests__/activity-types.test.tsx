import { delay, http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { TransactionHistoryCard } from "../TransactionHistoryCard";
import { executionID, run, topUp } from "./fixtures";

describe("Transaction history activity and states", () => {
  it.each([
    ["REFUND", -500, "Top-up refunded", "−$5.00"],
    ["SUBSCRIPTION", -2000, "Subscription payment", "−$20.00"],
    ["GRANT", 500, "Credits granted", "+$5.00"],
    ["CARD_CHECK", 0, "Card verification", "$0.00"],
    ["USAGE", -500, "Daily limit reset", "−$5.00"],
  ] as const)(
    "renders %s as a signed credit-balance change",
    async (type, amount, name, formatted) => {
      server.use(
        http.get("*/api/credits/transactions", () =>
          HttpResponse.json({
            transactions: [
              { ...topUp, transaction_type: type, description: name, amount },
            ],
            next_cursor: null,
          }),
        ),
      );
      render(<TransactionHistoryCard />);
      await screen.findByText(name);
      const money = screen.getByText(formatted);
      expect(money.classList.contains("text-red-600")).toBe(false);
      expect(money.classList.contains("text-green-700")).toBe(
        Number(amount) > 0,
      );
    },
  );

  it("renders direct block usage outside an agent run", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            {
              ...topUp,
              transaction_type: "USAGE",
              activity_type: "block_usage",
              description: "Block usage",
              amount: -12,
            },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    expect(await screen.findByText("Direct block usage")).toBeDefined();
    expect(screen.getByText("Outside an agent run")).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: /details for Direct block usage/i }),
    );
    expect(
      screen.getByText("A paid block call without an associated agent run."),
    ).toBeDefined();
  });

  it("uses the conversation destination for Autopilot tool use, never a library route", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            {
              ...run,
              activity_type: "copilot_tools",
              conversation_id: "chat-123",
              conversation_title: "Market landscape",
              library_agent_id: null,
              execution_available: false,
              execution_status: null,
            },
          ],
          next_cursor: null,
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    await screen.findByText("Autopilot tool use");
    expect(screen.getByText("Market landscape")).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: /details for Autopilot tool use/i }),
    );
    expect(
      screen
        .getByRole("link", { name: /View conversation/ })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=chat-123");
    expect(screen.queryByRole("link", { name: /View task/ })).toBeNull();
    expect(
      screen.getByText(/Subscription usage is tracked separately/),
    ).toBeDefined();
  });

  it("opens related receipts without adding child charges to the parent amount", async () => {
    const childID = "child-run";
    const child = {
      ...run,
      id: "execution:child-run",
      usage_execution_id: childID,
      agent_name: "Website summary",
      amount: -6,
      usage_charge_amount: -6,
      usage_fee_amount: 0,
      usage_adjustment_amount: 0,
      library_agent_id: null,
      parent_execution_id: executionID,
      parent_agent_name: "Morning briefing",
    };
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [
            {
              ...run,
              related_executions: [
                {
                  execution_id: childID,
                  agent_name: "Website summary",
                  amount: null,
                },
              ],
            },
            child,
          ],
          next_cursor: null,
          snapshot_at: "2026-09-05T08:14:10Z",
        }),
      ),
    );
    render(<TransactionHistoryCard />);
    await screen.findByText("Morning briefing");
    fireEvent.click(
      screen.getByRole("button", { name: /details for Morning briefing/i }),
    );
    const parentReceipt = screen.getByRole("region", {
      name: "Morning briefing credit receipt",
    });
    expect(within(parentReceipt).getByText("−$0.12")).toBeDefined();
    expect(within(parentReceipt).getByText("−$0.06")).toBeDefined();
    expect(within(parentReceipt).queryByText("−$0.18")).toBeNull();
    fireEvent.click(
      within(parentReceipt).getByRole("button", { name: "Website summary" }),
    );
    expect(
      screen.getByRole("region", { name: "Website summary credit receipt" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("region", { name: "Morning briefing credit receipt" }),
    ).toBeNull();
    expect(document.activeElement?.id).toBe(
      "transaction-details-execution:child-run",
    );
  });

  it("shows loading, then the genuine empty state", async () => {
    server.use(
      http.get("*/api/credits/transactions", async () => {
        await delay(100);
        return HttpResponse.json({ transactions: [], next_cursor: null });
      }),
    );
    render(<TransactionHistoryCard />);
    expect(
      screen.getByRole("status", { name: "Loading transaction history" }),
    ).toBeDefined();
    expect(await screen.findByText("No transactions yet.")).toBeDefined();
  });

  it("retries an initial error without claiming that ongoing credit activity stopped", async () => {
    let attempts = 0;
    server.use(
      http.get("*/api/credits/transactions", () => {
        attempts += 1;
        return attempts === 1
          ? HttpResponse.json({ detail: "Unavailable" }, { status: 500 })
          : HttpResponse.json({ transactions: [topUp], next_cursor: null });
      }),
    );
    render(<TransactionHistoryCard />);
    await screen.findByText(/We couldn.t load your transactions/);
    expect(screen.queryByText(/Your credit balance is unchanged/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));
    expect(await screen.findByText("Credits added")).toBeDefined();
    expect(document.activeElement?.textContent).toBe("Transaction history");
  });
});
