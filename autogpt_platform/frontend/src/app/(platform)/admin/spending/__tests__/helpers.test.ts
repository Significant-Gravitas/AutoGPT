import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import type { CopilotWeeklyUsageRow } from "@/app/api/__generated__/models/copilotWeeklyUsageRow";
import type { UserTransaction } from "@/app/api/__generated__/models/userTransaction";

import {
  buildCopilotUsageCsv,
  buildCreditTransactionsCsv,
  dateInputToUtcIso,
  dateInputToUtcIsoEnd,
  defaultEndDate,
  defaultStartDate,
  downloadCsv,
} from "../helpers";

function makeTx(overrides: Partial<UserTransaction> = {}): UserTransaction {
  return {
    transaction_key: "tx-1",
    transaction_time: "2026-04-01T10:00:00Z",
    transaction_type: "TOP_UP",
    amount: 5000,
    running_balance: 12500,
    current_balance: 12500,
    user_id: "user-1",
    user_email: "alice@example.com",
    reason: "Stripe checkout",
    admin_email: null,
    ...overrides,
  } as UserTransaction;
}

function makeUsage(
  overrides: Partial<CopilotWeeklyUsageRow> = {},
): CopilotWeeklyUsageRow {
  return {
    user_id: "user-1",
    user_email: "alice@example.com",
    week_start: "2026-03-30T00:00:00Z",
    week_end: "2026-04-05T23:59:59.999Z",
    copilot_cost_microdollars: 1_500_000,
    tier: "PRO",
    weekly_limit_microdollars: 25_000_000,
    percent_used: 6.0,
    ...overrides,
  } as CopilotWeeklyUsageRow;
}

describe("buildCreditTransactionsCsv", () => {
  test("emits header followed by one CRLF-terminated row per transaction", () => {
    const csv = buildCreditTransactionsCsv([makeTx()]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(2);
    expect(lines[0]).toBe(
      [
        '"transaction_id"',
        '"user_id"',
        '"user_email"',
        '"created_at"',
        '"type"',
        '"amount_usd"',
        '"running_balance_usd"',
        '"admin_email"',
        '"reason"',
      ].join(","),
    );
  });

  test("converts cents to USD with two decimals", () => {
    const csv = buildCreditTransactionsCsv([
      makeTx({ amount: 12345, running_balance: -678 }),
    ]);
    const row = csv.split("\r\n")[1];
    expect(row).toContain('"123.45"');
    expect(row).toContain('"-6.78"');
  });

  test("escapes embedded quotes by doubling them", () => {
    const csv = buildCreditTransactionsCsv([
      makeTx({ reason: 'has "quotes" inside' }),
    ]);
    expect(csv).toContain('"has ""quotes"" inside"');
  });

  test("emits empty string for null fields without breaking row shape", () => {
    const csv = buildCreditTransactionsCsv([
      makeTx({ user_email: null, admin_email: null, reason: null }),
    ]);
    const row = csv.split("\r\n")[1];
    expect(row.split(",")).toHaveLength(9);
    expect(row).toContain('""');
  });

  test("renders each transaction as its own row", () => {
    const csv = buildCreditTransactionsCsv([
      makeTx({ transaction_key: "tx-a" }),
      makeTx({ transaction_key: "tx-b" }),
    ]);
    expect(csv.split("\r\n")).toHaveLength(3);
  });
});

describe("buildCopilotUsageCsv", () => {
  test("converts microdollars to USD (6dp for cost, 2dp for limit)", () => {
    const csv = buildCopilotUsageCsv([
      makeUsage({
        copilot_cost_microdollars: 2_500_000,
        weekly_limit_microdollars: 25_000_000,
      }),
    ]);
    const row = csv.split("\r\n")[1];
    expect(row).toContain('"2.500000"');
    expect(row).toContain('"25.00"');
  });

  test("formats percent_used with two decimal places", () => {
    const csv = buildCopilotUsageCsv([makeUsage({ percent_used: 33.7 })]);
    expect(csv.split("\r\n")[1]).toContain('"33.70"');
  });

  test("emits the expected header row", () => {
    const csv = buildCopilotUsageCsv([]);
    expect(csv).toBe(
      [
        '"user_id"',
        '"user_email"',
        '"week_start"',
        '"week_end"',
        '"copilot_cost_usd"',
        '"tier"',
        '"weekly_limit_usd"',
        '"percent_used"',
      ].join(","),
    );
  });
});

describe("dateInputToUtcIso / dateInputToUtcIsoEnd", () => {
  test("returns null for an empty string", () => {
    expect(dateInputToUtcIso("")).toBeNull();
    expect(dateInputToUtcIsoEnd("")).toBeNull();
  });

  test("anchors start to UTC midnight of the input date", () => {
    expect(dateInputToUtcIso("2026-03-15")).toBe("2026-03-15T00:00:00.000Z");
  });

  test("anchors end to UTC end-of-day so the inclusive filter covers the day", () => {
    expect(dateInputToUtcIsoEnd("2026-03-15")).toBe("2026-03-15T23:59:59.999Z");
  });
});

describe("defaultStartDate / defaultEndDate", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("defaultStartDate returns 30 UTC days before now", () => {
    vi.setSystemTime(new Date("2026-04-30T12:34:56Z"));
    expect(defaultStartDate()).toBe("2026-03-31");
  });

  test("defaultEndDate returns today's UTC date string", () => {
    vi.setSystemTime(new Date("2026-04-30T23:59:00Z"));
    expect(defaultEndDate()).toBe("2026-04-30");
  });

  test("UTC arithmetic — late-night-PST viewer still gets correct UTC date", () => {
    // 2026-04-30 23:30 PST ⇒ 2026-05-01 06:30 UTC
    vi.setSystemTime(new Date("2026-05-01T06:30:00Z"));
    expect(defaultEndDate()).toBe("2026-05-01");
    expect(defaultStartDate()).toBe("2026-04-01");
  });
});

describe("downloadCsv", () => {
  test("clicks an anchor and defers URL.revokeObjectURL via setTimeout", () => {
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:mock");
    const revokeUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    vi.useFakeTimers();
    downloadCsv("col1\r\nval1", "out.csv");
    expect(createUrl).toHaveBeenCalledTimes(1);
    expect(anchorClick).toHaveBeenCalledTimes(1);
    // revoke is deferred via setTimeout so the download stream gets to start
    expect(revokeUrl).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(revokeUrl).toHaveBeenCalledWith("blob:mock");

    createUrl.mockRestore();
    revokeUrl.mockRestore();
    anchorClick.mockRestore();
    vi.useRealTimers();
  });
});
