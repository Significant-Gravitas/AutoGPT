import { describe, expect, it } from "vitest";
import {
  asObject,
  dictToOutputItems,
  formatBytes,
  formatWhen,
  inline,
  integrationIconSrc,
  safeHostname,
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
});
