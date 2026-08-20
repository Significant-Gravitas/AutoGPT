import { describe, expect, it } from "vitest";
import {
  asItems,
  asObject,
  dictToOutputItems,
  formatBytes,
  formatWhen,
  humanizeKey,
  inline,
  integrationIconSrc,
  resultItemKey,
  safeHostname,
  str,
  stripBaseFields,
} from "../resultHelpers";

describe("asObject", () => {
  it("accepts records and JSON-encoded records", () => {
    expect(asObject({ ok: true })).toEqual({ ok: true });
    expect(asObject('{"ok":true}')).toEqual({ ok: true });
  });

  it("rejects arrays, including JSON-encoded arrays", () => {
    expect(asObject(["value"])).toBeNull();
    expect(asObject('["value"]')).toBeNull();
  });

  it("rejects malformed JSON and non-object primitives", () => {
    expect(asObject("not json")).toBeNull();
    expect(asObject(null)).toBeNull();
    expect(asObject(42)).toBeNull();
    expect(asObject("42")).toBeNull();
  });
});

describe("str", () => {
  it("returns the first non-empty string across candidate keys", () => {
    expect(str({ a: "", b: "   ", c: "value" }, "a", "b", "c")).toBe("value");
  });

  it("skips non-string values and returns null when nothing matches", () => {
    expect(str({ a: 3, b: true }, "a", "b")).toBeNull();
    expect(str({}, "missing")).toBeNull();
  });
});

describe("asItems", () => {
  it("returns null for non-arrays and empty arrays", () => {
    expect(asItems({ length: 1 })).toBeNull();
    expect(asItems([])).toBeNull();
    expect(asItems("nope")).toBeNull();
  });

  it("wraps primitive entries as value records", () => {
    expect(asItems(["a", { id: "1" }])).toEqual([{ value: "a" }, { id: "1" }]);
  });
});

describe("stripBaseFields", () => {
  it("removes the response envelope fields only", () => {
    expect(
      stripBaseFields({
        type: "done",
        message: "ok",
        session_id: "s1",
        result: 5,
      }),
    ).toEqual({ result: 5 });
  });
});

describe("resultItemKey", () => {
  it("prefers the highest-priority identifying field", () => {
    expect(resultItemKey({ name: "n", id: "abc" }, 0)).toBe("id:abc");
    expect(resultItemKey({ url: "https://x", title: "t" }, 0)).toBe(
      "url:https://x",
    );
    expect(resultItemKey({ execution_id: 7 }, 0)).toBe("execution_id:7");
  });

  it("falls back to the index with inline content", () => {
    expect(resultItemKey({ other: true }, 3)).toBe('item:3:{"other":true}');
  });
});

describe("humanizeKey", () => {
  it("replaces underscores and capitalizes the first letter", () => {
    expect(humanizeKey("status_code")).toBe("Status code");
    expect(humanizeKey("ok")).toBe("Ok");
  });
});

describe("safeHostname", () => {
  it("returns a display hostname for valid URLs", () => {
    expect(safeHostname("https://www.example.com/path")).toBe("example.com");
  });

  it("returns null for invalid URLs", () => {
    expect(safeHostname("not a URL")).toBeNull();
  });
});

describe("integrationIconSrc", () => {
  it("normalizes allowed provider characters", () => {
    expect(integrationIconSrc("Google Maps")).toBe(
      "/integrations/google_maps.png",
    );
    expect(integrationIconSrc("../../Google Maps?<script>")).toBe(
      "/integrations/google_mapsscript.png",
    );
  });

  it("rejects providers without safe characters", () => {
    expect(integrationIconSrc("../../")).toBeNull();
  });
});

describe("result formatting", () => {
  it("flattens single-value block outputs", () => {
    expect(dictToOutputItems({ result: ["done"], count: [1, 2] })).toEqual([
      { name: "result", value: "done" },
      { name: "count", value: [1, 2] },
    ]);
  });

  it("truncates long inline JSON values", () => {
    expect(inline({ value: "x".repeat(150) })).toHaveLength(121);
    expect(inline({ value: "x".repeat(150) })).toMatch(/…$/);
  });

  it("formats byte units", () => {
    expect(formatBytes(12)).toBe("12 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1572864)).toBe("1.5 MB");
  });

  it("returns invalid date values unchanged", () => {
    expect(formatWhen("not-a-date")).toBe("not-a-date");
  });

  it("formats valid dates for display", () => {
    const formatted = formatWhen("2026-08-21T10:00:00Z");
    expect(formatted).not.toBe("2026-08-21T10:00:00Z");
    expect(formatted).toMatch(/\d/);
  });

  it("returns null for empty output dictionaries", () => {
    expect(dictToOutputItems({})).toBeNull();
    expect(dictToOutputItems(["a"])).toBeNull();
    expect(dictToOutputItems("text")).toBeNull();
  });

  it("stringifies inline primitives", () => {
    expect(inline("text")).toBe("text");
    expect(inline(3)).toBe("3");
    expect(inline(false)).toBe("false");
  });
});
