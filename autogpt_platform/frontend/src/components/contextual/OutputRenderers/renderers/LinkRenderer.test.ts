import { describe, expect, it } from "vitest";
import { linkRenderer } from "./LinkRenderer";

describe("LinkRenderer canRender", () => {
  it("detects plain HTTP URLs", () => {
    expect(linkRenderer.canRender("https://example.com")).toBe(true);
    expect(linkRenderer.canRender("http://example.com/page")).toBe(true);
    expect(linkRenderer.canRender("https://example.com/path/to/resource")).toBe(
      true,
    );
  });

  it("detects URLs with query params", () => {
    expect(linkRenderer.canRender("https://example.com/search?q=test")).toBe(
      true,
    );
  });

  it("rejects non-URL strings", () => {
    expect(linkRenderer.canRender("just some text")).toBe(false);
    expect(linkRenderer.canRender("not a url at all")).toBe(false);
    expect(linkRenderer.canRender("")).toBe(false);
  });

  it("rejects non-string values", () => {
    expect(linkRenderer.canRender(123)).toBe(false);
    expect(linkRenderer.canRender(null)).toBe(false);
    expect(linkRenderer.canRender({ url: "https://example.com" })).toBe(false);
  });

  it("rejects data URIs (handled by other renderers)", () => {
    expect(linkRenderer.canRender("data:image/png;base64,abc")).toBe(false);
  });

  it("rejects workspace URIs (handled by WorkspaceFileRenderer)", () => {
    expect(linkRenderer.canRender("workspace://file-123#image/png")).toBe(
      false,
    );
  });
});
