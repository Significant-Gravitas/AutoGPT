import { describe, expect, it } from "vitest";
import { TAILWIND_CDN_URL, wrapWithHeadInjection } from "../iframe-sandbox-csp";

describe("wrapWithHeadInjection", () => {
  const injection = '<script src="https://example.com/lib.js"></script>';

  it("injects after <head> when document has a head tag", () => {
    const html = "<html><head><title>Test</title></head><body>Hi</body></html>";
    const result = wrapWithHeadInjection(html, injection);
    expect(result).toContain(`<head>${injection}<title>Test</title>`);
  });

  it("injects after <head> with attributes", () => {
    const html = '<html><head lang="en"><title>Test</title></head></html>';
    const result = wrapWithHeadInjection(html, injection);
    expect(result).toContain(`<head lang="en">${injection}<title>Test</title>`);
  });

  it("is case-insensitive for head tag", () => {
    const html = "<HTML><HEAD><TITLE>Test</TITLE></HEAD></HTML>";
    const result = wrapWithHeadInjection(html, injection);
    expect(result).toContain(`<HEAD>${injection}<TITLE>`);
  });

  it("wraps headless content in a full document skeleton", () => {
    const html = "<div>Just a fragment</div>";
    const result = wrapWithHeadInjection(html, injection);
    expect(result).toBe(
      `<!doctype html><html><head>${injection}</head><body>${html}</body></html>`,
    );
  });

  it("wraps empty string in a skeleton", () => {
    const result = wrapWithHeadInjection("", injection);
    expect(result).toContain("<head>" + injection + "</head>");
    expect(result).toContain("<body></body>");
  });
});

describe("TAILWIND_CDN_URL", () => {
  it("is pinned to a specific version", () => {
    expect(TAILWIND_CDN_URL).toMatch(
      /^https:\/\/cdn\.tailwindcss\.com\/\d+\.\d+\.\d+$/,
    );
  });
});

describe("no CSP is exported", () => {
  it("does not export ARTIFACT_IFRAME_CSP", async () => {
    const mod = await import("../iframe-sandbox-csp");
    expect("ARTIFACT_IFRAME_CSP" in mod).toBe(false);
  });

  it("does not export cspMetaTag", async () => {
    const mod = await import("../iframe-sandbox-csp");
    expect("cspMetaTag" in mod).toBe(false);
  });
});
