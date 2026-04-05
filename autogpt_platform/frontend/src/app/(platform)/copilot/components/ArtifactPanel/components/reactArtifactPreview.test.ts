import { describe, expect, it } from "vitest";
import { escapeHtml } from "./reactArtifactPreview";

describe("escapeHtml", () => {
  it("escapes &, <, >, \", '", () => {
    expect(escapeHtml("a & b")).toBe("a &amp; b");
    expect(escapeHtml("<script>")).toBe("&lt;script&gt;");
    expect(escapeHtml('hello "world"')).toBe("hello &quot;world&quot;");
    expect(escapeHtml("it's")).toBe("it&#39;s");
  });

  it("neutralizes a </title> escape attempt", () => {
    // Used to escape a title that lands inside <title>${safeTitle}</title>
    const out = escapeHtml("</title><script>alert(1)</script>");
    expect(out).not.toContain("</title>");
    expect(out).not.toContain("<script>");
    expect(out).toContain("&lt;/title&gt;");
    expect(out).toContain("&lt;script&gt;");
  });

  it("escapes ampersand first so entities aren't double-escaped in the wrong order", () => {
    // If & were escaped AFTER <, the < → &lt; output would become &amp;lt;.
    // Verify the & substitution ran on the raw input only.
    expect(escapeHtml("A&B<C")).toBe("A&amp;B&lt;C");
  });

  it("is safe on empty / plain strings", () => {
    expect(escapeHtml("")).toBe("");
    expect(escapeHtml("plain text 123")).toBe("plain text 123");
  });
});
