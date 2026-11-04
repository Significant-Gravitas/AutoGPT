import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderToString } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { TransactionHistoryCard } from "../TransactionHistoryCard";

describe("Transaction history time zone", () => {
  it("keeps initial markup stable across server and browser time zones", () => {
    const options = Intl.DateTimeFormat().resolvedOptions();
    const resolvedOptions = vi.spyOn(
      Intl.DateTimeFormat.prototype,
      "resolvedOptions",
    );
    const card = (
      <QueryClientProvider client={new QueryClient()}>
        <TransactionHistoryCard />
      </QueryClientProvider>
    );

    try {
      resolvedOptions.mockReturnValue({ ...options, timeZone: "UTC" });
      const serverMarkup = renderToString(card);
      resolvedOptions.mockReturnValue({ ...options, timeZone: "Europe/Paris" });
      const browserMarkup = renderToString(card);

      expect(browserMarkup).toBe(serverMarkup);
      expect(serverMarkup).toContain("USD · Your local time");
    } finally {
      resolvedOptions.mockRestore();
    }
  });
});
