import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { htmlRenderer } from "./HTMLRenderer";

describe("HTMLRenderer", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders text/html content in a sandboxed iframe", () => {
    const { container } = render(
      <>
        {htmlRenderer.render("<h1>Hi</h1>", {
          mimeType: "text/html",
          filename: "page.html",
        })}
      </>,
    );
    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute("sandbox")).toBe("allow-scripts");
  });

  it("injects the fragment-link interceptor into the srcDoc (regression)", () => {
    const { container } = render(
      <>
        {htmlRenderer.render(
          '<html><head></head><body><a href="#x">x</a><div id="x">x</div></body></html>',
          { mimeType: "text/html", filename: "page.html" },
        )}
      </>,
    );
    const srcdoc = container.querySelector("iframe")?.getAttribute("srcdoc");
    expect(srcdoc).toBeTruthy();
    expect(srcdoc).toContain("__fragmentLinkInterceptor");
    expect(srcdoc).toContain('a[href^="#"]');
    expect(srcdoc).toContain("scrollIntoView");
  });

  it("canRender recognises text/html mime type and .html/.htm filenames", () => {
    expect(
      htmlRenderer.canRender("<h1>Hi</h1>", { mimeType: "text/html" }),
    ).toBe(true);
    expect(
      htmlRenderer.canRender("<h1>Hi</h1>", { filename: "report.html" }),
    ).toBe(true);
    expect(
      htmlRenderer.canRender("<h1>Hi</h1>", { filename: "report.htm" }),
    ).toBe(true);
    expect(
      htmlRenderer.canRender("<h1>Hi</h1>", { mimeType: "text/plain" }),
    ).toBe(false);
  });
});
