import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  desktopStreamRenderer,
  DesktopStreamPreview,
} from "../DesktopStreamRenderer";

const streamValue = {
  kind: "desktop_stream" as const,
  url: "https://6080-sandbox.e2b.app/vnc.html?autoconnect=true",
  provider: "e2b",
  sandbox_id: "sbx-123",
  requires_auth: false,
};

describe("DesktopStreamRenderer", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("canRender matches desktop_stream objects only", () => {
    expect(desktopStreamRenderer.canRender(streamValue)).toBe(true);
    expect(desktopStreamRenderer.canRender({ kind: "other" })).toBe(false);
    expect(desktopStreamRenderer.canRender("https://example.com")).toBe(false);
    expect(desktopStreamRenderer.canRender(null)).toBe(false);
    expect(
      desktopStreamRenderer.canRender({ kind: "desktop_stream", url: 42 }),
    ).toBe(false);
  });

  it("refuses an absolute link to our own origin", () => {
    vi.stubGlobal("location", new URL("https://app.test/copilot"));
    expect(
      desktopStreamRenderer.canRender({
        ...streamValue,
        url: "https://app.test/library",
      }),
    ).toBe(false);
    expect(desktopStreamRenderer.canRender(streamValue)).toBe(true);
  });

  it("refuses anything but an https URL or the desktop-preview link", () => {
    for (const url of [
      "javascript:alert(document.cookie)",
      "http://6080-sandbox.e2b.app/vnc.html",
      "data:text/html,<script>1</script>",
      "not a url",
      "//evil.example/vnc.html",
      "/api/proxy/api/v1/items",
      "/api/proxy/api/desktop-preview/../v1/items",
      "/library",
    ]) {
      expect(desktopStreamRenderer.canRender({ ...streamValue, url })).toBe(
        false,
      );
    }
    // The owner-bound preview link is root-relative so it works on any
    // origin the owner is signed in to, plain-http local stacks included.
    expect(
      desktopStreamRenderer.canRender({
        ...streamValue,
        url: "/api/proxy/api/desktop-preview?token=abc",
        requires_auth: true,
      }),
    ).toBe(true);
  });

  it("renders an interactive iframe pointing at the stream URL", () => {
    const { container } = render(
      <>{desktopStreamRenderer.render(streamValue)}</>,
    );
    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute("src")).toBe(streamValue.url);
    expect(iframe?.getAttribute("sandbox")).toBe(
      "allow-scripts allow-same-origin allow-popups",
    );
    expect(iframe?.getAttribute("allow")).toBe(
      "clipboard-read; clipboard-write; fullscreen",
    );
    expect(screen.getByText("Interactive Desktop")).toBeDefined();
    expect(screen.getByText("e2b")).toBeDefined();
  });

  it("provides an open-in-new-tab link", () => {
    render(<>{desktopStreamRenderer.render(streamValue)}</>);
    const link = screen.getByRole("link", { name: /open in new tab/i });
    expect(link.getAttribute("href")).toBe(streamValue.url);
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it.each(["body", "body-md"] as const)(
    "preserves the %s text size for its callers",
    (textSize) => {
      const { rerender } = render(
        <DesktopStreamPreview value={streamValue} textSize={textSize} />,
      );
      expect(
        screen
          .getByText("Interactive Desktop")
          .classList.contains(`text-${textSize}`),
      ).toBe(true);

      rerender(
        <DesktopStreamPreview
          value={{ ...streamValue, requires_auth: true }}
          ownerView={false}
          textSize={textSize}
        />,
      );
      expect(
        screen
          .getByText(/only visible to the owner/i)
          .classList.contains(`text-${textSize}`),
      ).toBe(true);
    },
  );

  it("tells the viewer the desktop is shared with the AI, and owner-only when auth is required", () => {
    render(<>{desktopStreamRenderer.render(streamValue)}</>);
    expect(screen.getByText(/visible to it/i)).toBeDefined();
    expect(screen.queryByText(/only the owner/i)).toBeNull();
    cleanup();
    render(
      <>
        {desktopStreamRenderer.render({ ...streamValue, requires_auth: true })}
      </>,
    );
    expect(screen.getByText(/only the owner/i)).toBeDefined();
  });

  it("copies the stream URL", () => {
    const copy = desktopStreamRenderer.getCopyContent(streamValue);
    expect(copy?.data).toBe(streamValue.url);
  });

  it("has no download content", () => {
    expect(desktopStreamRenderer.getDownloadContent(streamValue)).toBeNull();
  });
});
