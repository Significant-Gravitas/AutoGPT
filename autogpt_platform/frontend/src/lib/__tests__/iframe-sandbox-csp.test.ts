import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  FRAGMENT_LINK_INTERCEPTOR_SCRIPT,
  TAILWIND_CDN_URL,
  wrapWithHeadInjection,
} from "../iframe-sandbox-csp";

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

describe("FRAGMENT_LINK_INTERCEPTOR_SCRIPT", () => {
  // Evaluate the script body (without <script> tags) against the current
  // document. Because sandboxed srcdoc iframes run their scripts in isolation
  // anyway, the behavior we care about is just "this code, when executed in
  // a document, intercepts #anchor clicks and calls scrollIntoView".
  //
  // Parse the exported <script> via the DOM rather than regex — CodeQL flags
  // regex-based HTML stripping, and the test already runs in a DOM env.
  function installInterceptor() {
    const template = document.createElement("template");
    template.innerHTML = FRAGMENT_LINK_INTERCEPTOR_SCRIPT;
    const script = template.content.querySelector("script");
    if (!script) throw new Error("Interceptor script tag not found");
    new Function(script.textContent ?? "")();
  }

  let cleanup: (() => void) | null = null;

  beforeEach(() => {
    document.body.innerHTML = "";
  });

  afterEach(() => {
    if (cleanup) cleanup();
    cleanup = null;
    document.body.innerHTML = "";
    const doc = document as Document & {
      __fragmentLinkInterceptor?: EventListener;
    };
    if (doc.__fragmentLinkInterceptor) {
      document.removeEventListener("click", doc.__fragmentLinkInterceptor);
      delete doc.__fragmentLinkInterceptor;
    }
  });

  it("exports a <script> tag wrapping the interceptor", () => {
    expect(FRAGMENT_LINK_INTERCEPTOR_SCRIPT.startsWith("<script>")).toBe(true);
    expect(FRAGMENT_LINK_INTERCEPTOR_SCRIPT.endsWith("</script>")).toBe(true);
    expect(FRAGMENT_LINK_INTERCEPTOR_SCRIPT).toContain("addEventListener");
    expect(FRAGMENT_LINK_INTERCEPTOR_SCRIPT).toContain("scrollIntoView");
    expect(FRAGMENT_LINK_INTERCEPTOR_SCRIPT).toContain('a[href^="#"]');
  });

  // Install the interceptor first, then a tail listener that records
  // defaultPrevented. Listeners fire in registration order, so the tail
  // sees the post-interceptor state.
  function installWithObserver() {
    installInterceptor();
    const observed = { defaulted: false };
    const listener = (e: Event) => {
      observed.defaulted = e.defaultPrevented;
    };
    document.addEventListener("click", listener);
    cleanup = () => document.removeEventListener("click", listener);
    return observed;
  }

  it("intercepts fragment-link clicks, calls preventDefault, and scrolls the target into view", () => {
    document.body.innerHTML = `
      <nav><a id="nav-link" href="#activation">Activation</a></nav>
      <section id="activation">Target</section>
    `;
    const scrollSpy = vi.fn();
    document.getElementById("activation")!.scrollIntoView = scrollSpy;

    const observed = installWithObserver();

    document.getElementById("nav-link")!.click();

    expect(scrollSpy).toHaveBeenCalledTimes(1);
    expect(observed.defaulted).toBe(true);
  });

  it("does not intercept bare '#' links (no target id)", () => {
    document.body.innerHTML = `<a id="top" href="#">Back to top</a>`;
    const observed = installWithObserver();

    document.getElementById("top")!.click();

    expect(observed.defaulted).toBe(false);
  });

  it("does not intercept links with no matching target in the document", () => {
    document.body.innerHTML = `<a id="dangle" href="#missing">Nowhere</a>`;
    const observed = installWithObserver();

    document.getElementById("dangle")!.click();

    expect(observed.defaulted).toBe(false);
  });

  it("does not intercept non-fragment links", () => {
    document.body.innerHTML = `<a id="ext" href="https://example.com/x">Ext</a>`;
    installInterceptor();
    const observed = { defaulted: false };
    const listener = (e: Event) => {
      observed.defaulted = e.defaultPrevented;
      e.preventDefault();
    };
    document.addEventListener("click", listener);
    cleanup = () => document.removeEventListener("click", listener);

    document.getElementById("ext")!.click();

    expect(observed.defaulted).toBe(false);
  });

  it("scrolls to target when click originates from a nested child of the anchor", () => {
    document.body.innerHTML = `
      <a id="outer" href="#costs"><span id="inner">💰 Costs</span></a>
      <section id="costs">Target</section>
    `;
    const scrollSpy = vi.fn();
    document.getElementById("costs")!.scrollIntoView = scrollSpy;

    installInterceptor();
    document.getElementById("inner")!.click();

    expect(scrollSpy).toHaveBeenCalledTimes(1);
  });

  it("handles percent-encoded ids", () => {
    document.body.innerHTML = `
      <a id="enc" href="#top%20costs">Jump</a>
      <section id="top costs">Target</section>
    `;
    const scrollSpy = vi.fn();
    document.getElementById("top costs")!.scrollIntoView = scrollSpy;

    installInterceptor();
    document.getElementById("enc")!.click();

    expect(scrollSpy).toHaveBeenCalledTimes(1);
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
