import { describe, expect, it } from "vitest";
import { csvRenderer } from "./CSVRenderer";

function downloadText(value: string, filename = "t.csv"): string {
  const dl = csvRenderer.getDownloadContent?.(value, { filename });
  if (!dl) throw new Error("no download content");
  return dl.filename;
}

describe("csvRenderer.canRender", () => {
  it("matches CSV mime type", () => {
    expect(csvRenderer.canRender("a,b\n1,2", { mimeType: "text/csv" })).toBe(
      true,
    );
  });
  it("matches .csv filename case-insensitively", () => {
    expect(csvRenderer.canRender("a,b", { filename: "data.CSV" })).toBe(true);
  });
  it("rejects non-string values", () => {
    expect(csvRenderer.canRender(42, { mimeType: "text/csv" })).toBe(false);
  });
  it("rejects strings without CSV hint", () => {
    expect(csvRenderer.canRender("a,b,c", {})).toBe(false);
  });
});

describe("csvRenderer.getDownloadContent", () => {
  it("uses filename from metadata", () => {
    expect(downloadText("a,b\n1,2", "my.csv")).toBe("my.csv");
  });
  it("falls back to data.csv", () => {
    const dl = csvRenderer.getDownloadContent?.("a,b\n1,2");
    expect(dl?.filename).toBe("data.csv");
  });
});

describe("csvRenderer.getCopyContent", () => {
  it("round-trips content as plain text", () => {
    const result = csvRenderer.getCopyContent?.("x,y\n1,2");
    expect(result?.mimeType).toBe("text/plain");
    expect(result?.data).toBe("x,y\n1,2");
  });
});

describe("csvRenderer.render (parse via render output smoke)", () => {
  // The parser itself isn't exported, so we exercise it through render.
  // These tests ensure render() doesn't throw on edge-case CSVs.
  it("handles empty input", () => {
    expect(() => csvRenderer.render("")).not.toThrow();
  });
  it("handles embedded newline inside quoted field", () => {
    const csv = 'name,bio\n"Alice","line1\nline2"\n"Bob","x"';
    expect(() => csvRenderer.render(csv)).not.toThrow();
  });
  it("strips BOM from first header cell (smoke)", () => {
    const csv = "\ufefftitle,count\nfoo,1";
    expect(() => csvRenderer.render(csv)).not.toThrow();
  });
  it("handles CRLF line endings", () => {
    const csv = "a,b\r\n1,2\r\n3,4";
    expect(() => csvRenderer.render(csv)).not.toThrow();
  });
  it("handles escaped double quote inside a quoted field", () => {
    const csv = 'name\n"She said ""hi"""';
    expect(() => csvRenderer.render(csv)).not.toThrow();
  });
});
