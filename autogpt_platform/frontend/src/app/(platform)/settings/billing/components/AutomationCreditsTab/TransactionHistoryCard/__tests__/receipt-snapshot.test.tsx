import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { TransactionHistoryCard } from "../TransactionHistoryCard";
import { run, topUp } from "./fixtures";

describe("Transaction history snapshot", () => {
  it("labels the credit cutoff even when a later page reports a currently completed run", async () => {
    const snapshot = "2026-09-05T08:14:10Z";
    server.use(
      http.get("*/api/credits/transactions", ({ request }) => {
        const laterPage = new URL(request.url).searchParams.has("cursor");
        return HttpResponse.json({
          transactions: laterPage
            ? [{ ...run, execution_status: "COMPLETED" }]
            : [topUp],
          next_cursor: laterPage ? null : "older-page",
          snapshot_at: snapshot,
        });
      }),
    );
    render(<TransactionHistoryCard />);
    await screen.findByText("Credits added");
    fireEvent.click(screen.getByRole("button", { name: /Load more/ }));
    await screen.findByText("Morning briefing");
    fireEvent.click(
      screen.getByRole("button", { name: /details for Morning briefing/i }),
    );
    const receipt = screen.getByRole("region", {
      name: "Morning briefing credit receipt",
    });
    expect(
      within(receipt).getByText("Run status").closest("div")?.textContent,
    ).toContain("Completed");
    const cutoff = within(receipt).getByText("Credits as of").closest("div");
    expect(cutoff?.querySelector("[title]")?.getAttribute("title")).toBe(
      new Date(snapshot).toLocaleString(undefined, {
        dateStyle: "long",
        timeStyle: "long",
      }),
    );
    expect(
      within(receipt).getByText(
        "Recorded charges and adjustments for this run.",
      ),
    ).toBeDefined();
    expect(within(receipt).queryByText(/All recorded/)).toBeNull();
  });
});
