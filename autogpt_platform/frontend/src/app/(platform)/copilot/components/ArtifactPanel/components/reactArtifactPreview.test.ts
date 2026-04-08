import { describe, expect, it } from "vitest";
import {
  buildReactArtifactSrcDoc,
  collectPreviewStyles,
  escapeHtml,
} from "./reactArtifactPreview";

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

describe("buildReactArtifactSrcDoc", () => {
  const STYLES = collectPreviewStyles();

  it("does not contain a CSP meta tag (see iframe-sandbox-csp.ts)", () => {
    const doc = buildReactArtifactSrcDoc("module.exports = {};", "A", STYLES);
    expect(doc).not.toContain("Content-Security-Policy");
  });

  it("includes SRI-pinned React and ReactDOM bundles", () => {
    const doc = buildReactArtifactSrcDoc("module.exports = {};", "A", STYLES);
    expect(doc).toContain(
      'src="https://unpkg.com/react@18.3.1/umd/react.production.min.js"',
    );
    expect(doc).toContain('integrity="sha384-');
    expect(doc).toContain(
      'src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js"',
    );
  });

  it("escapes the title into the <title> tag", () => {
    const doc = buildReactArtifactSrcDoc(
      "module.exports = {};",
      "</title><script>alert(1)</script>",
      STYLES,
    );
    expect(doc).not.toMatch(/<title><\/title><script>/);
    expect(doc).toContain("&lt;/title&gt;");
  });

  it("escapes </script> sequences in compiled code so the inline script can't be broken out of", () => {
    // A legitimate artifact may contain the literal string "</script>" inside
    // a JSX template or string; it must be \u003c-escaped before embedding.
    const compiled = 'const x = "</script><script>alert(1)</script>";';
    const doc = buildReactArtifactSrcDoc(compiled, "A", STYLES);
    // The raw compiled string should NOT appear verbatim inside the srcDoc
    // (that would break out of the runtime <script>).
    expect(doc).not.toContain('"</script><script>alert(1)</script>"');
    // Instead, the escaped \u003c/script> form is what we expect.
    expect(doc).toContain("\\u003c/script>");
  });

  it("wires up #root and #error containers", () => {
    const doc = buildReactArtifactSrcDoc("module.exports = {};", "A", STYLES);
    expect(doc).toContain('<div id="root">');
    expect(doc).toContain('<div id="error">');
  });

  it("injects the styles markup supplied by collectPreviewStyles", () => {
    const doc = buildReactArtifactSrcDoc("module.exports = {};", "A", STYLES);
    expect(doc).toContain("box-sizing: border-box");
  });
});
