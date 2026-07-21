import { describe, expect, test, vi } from "vitest";
import {
  buildEstimatesJson,
  dateInputToUtcIso,
  dateInputToUtcIsoEnd,
  defaultEndDate,
  defaultStartDate,
  downloadJson,
} from "../helpers";

describe("dateInputToUtcIso", () => {
  test("returns null on empty input", () => {
    expect(dateInputToUtcIso("")).toBeNull();
  });

  test("converts YYYY-MM-DD to ISO 8601 UTC midnight", () => {
    expect(dateInputToUtcIso("2026-05-07")).toBe("2026-05-07T00:00:00.000Z");
  });
});

describe("dateInputToUtcIsoEnd", () => {
  test("returns null on empty input", () => {
    expect(dateInputToUtcIsoEnd("")).toBeNull();
  });

  test("pins to end-of-day so the inclusive end filter covers the whole day", () => {
    expect(dateInputToUtcIsoEnd("2026-05-07")).toBe("2026-05-07T23:59:59.999Z");
  });
});

describe("defaultStartDate", () => {
  test("subtracts 6 days so the inclusive [start, end] window is exactly 7 days", () => {
    // Mock Date so the assertion is stable regardless of when CI runs.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-07T12:00:00Z"));
    try {
      expect(defaultStartDate()).toBe("2026-05-01");
      expect(defaultEndDate()).toBe("2026-05-07");
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("buildEstimatesJson", () => {
  test("serialises rows into the JSON file shape with stable key order", () => {
    const json = buildEstimatesJson(
      [
        {
          block_id: "blk-1",
          block_name: "Foo",
          cost_type: "second",
          samples: 10,
          mean_credits: 7,
          p50_credits: 6,
          p95_credits: 12,
        },
      ],
      "2026-05-07T13:00:00.000Z",
      7,
    );
    const parsed = JSON.parse(json);
    expect(parsed.version).toBe(1);
    expect(parsed.generated_at).toBe("2026-05-07T13:00:00.000Z");
    expect(parsed.source_window_days).toBe(7);
    expect(parsed.estimates["blk-1"]).toEqual({
      block_name: "Foo",
      cost_type: "second",
      samples: 10,
      mean_credits: 7,
    });
    // p50/p95 don't enter the runtime JSON — they're for the admin UI only.
    expect(parsed.estimates["blk-1"].p50_credits).toBeUndefined();
    expect(json.endsWith("\n")).toBe(true);
  });

  test("emits an empty estimates object for zero rows", () => {
    const json = buildEstimatesJson([], "2026-05-07T13:00:00.000Z", 7);
    expect(JSON.parse(json).estimates).toEqual({});
  });
});

describe("downloadJson", () => {
  test("creates a Blob URL, triggers a click, and revokes the URL", async () => {
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:mock");
    const revokeUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    try {
      downloadJson('{"hello":"world"}\n', "test.json");

      expect(createUrl).toHaveBeenCalledTimes(1);
      expect(click).toHaveBeenCalledTimes(1);
      // The setTimeout(0) revoke fires on the next macrotask — flush it so the
      // assertion is deterministic and the test name's promise is honored.
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(revokeUrl).toHaveBeenCalledWith("blob:mock");
    } finally {
      // Failure-safe restore: spies survive even if the assertions above throw.
      createUrl.mockRestore();
      revokeUrl.mockRestore();
      click.mockRestore();
    }
  });
});
