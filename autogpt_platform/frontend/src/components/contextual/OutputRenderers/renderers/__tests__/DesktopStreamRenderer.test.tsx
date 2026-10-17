import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
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
    vi.restoreAllMocks();
    Reflect.deleteProperty(document, "fullscreenEnabled");
    Reflect.deleteProperty(HTMLIFrameElement.prototype, "requestFullscreen");
  });

  function mockFullscreenEnabled(enabled: boolean) {
    Object.defineProperty(document, "fullscreenEnabled", {
      configurable: true,
      get: () => enabled,
    });
  }

  function mockRequestFullscreen(impl: () => Promise<void>) {
    const spy = vi.fn(impl);
    Object.defineProperty(HTMLIFrameElement.prototype, "requestFullscreen", {
      configurable: true,
      writable: true,
      value: spy,
    });
    return spy;
  }

  it("hides the fullscreen button where the Fullscreen API is unsupported", () => {
    mockFullscreenEnabled(false);
    render(<DesktopStreamPreview value={streamValue} />);
    expect(screen.queryByRole("button", { name: /fullscreen/i })).toBeNull();
    expect(
      screen.getByRole("link", { name: /open in new tab/i }),
    ).toBeDefined();
  });

  it("requests fullscreen on the frame when supported", async () => {
    mockFullscreenEnabled(true);
    const requestFullscreen = mockRequestFullscreen(() => Promise.resolve());
    render(<DesktopStreamPreview value={streamValue} />);
    const button = await screen.findByRole("button", { name: /fullscreen/i });
    fireEvent.click(button);
    await waitFor(() => expect(requestFullscreen).toHaveBeenCalledOnce());
    expect(requestFullscreen.mock.instances[0]).toBeInstanceOf(
      HTMLIFrameElement,
    );
  });

  it("does not throw when the frame lacks requestFullscreen", async () => {
    mockFullscreenEnabled(true);
    Object.defineProperty(HTMLIFrameElement.prototype, "requestFullscreen", {
      configurable: true,
      writable: true,
      value: undefined,
    });
    render(<DesktopStreamPreview value={streamValue} />);
    const button = await screen.findByRole("button", { name: /fullscreen/i });
    expect(() => fireEvent.click(button)).not.toThrow();
  });

  it("swallows a rejected fullscreen request", async () => {
    mockFullscreenEnabled(true);
    const requestFullscreen = mockRequestFullscreen(() =>
      Promise.reject(new Error("denied")),
    );
    render(<DesktopStreamPreview value={streamValue} />);
    const button = await screen.findByRole("button", { name: /fullscreen/i });
    fireEvent.click(button);
    await waitFor(() => expect(requestFullscreen).toHaveBeenCalledOnce());
    expect(screen.getByText("Interactive Desktop")).toBeDefined();
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

  it("refuses anything but an https URL or our own origin-relative link", () => {
    for (const url of [
      "javascript:alert(document.cookie)",
      "http://6080-sandbox.e2b.app/vnc.html",
      "data:text/html,<script>1</script>",
      "not a url",
      "//evil.example/vnc.html",
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

  it("shows a notice instead of the frame to a viewer who is not the owner", () => {
    const ownerBound = {
      ...streamValue,
      url: "/api/proxy/api/desktop-preview?token=abc",
      requires_auth: true,
    };
    const { container } = render(
      <DesktopStreamPreview value={ownerBound} ownerView={false} />,
    );
    expect(container.querySelector("iframe")).toBeNull();
    expect(screen.getByText(/only visible to the owner/i)).toBeDefined();
    expect(screen.queryByRole("link", { name: /open in new tab/i })).toBeNull();
  });

  it("renders an interactive iframe pointing at the stream URL", () => {
    const { container } = render(
      <>{desktopStreamRenderer.render(streamValue)}</>,
    );
    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute("src")).toBe(streamValue.url);
    expect(iframe?.getAttribute("sandbox")).toContain("allow-scripts");
    expect(screen.getByText("Interactive Desktop")).toBeDefined();
    expect(screen.getByText("e2b")).toBeDefined();
  });

  it("provides an open-in-new-tab link", () => {
    render(<>{desktopStreamRenderer.render(streamValue)}</>);
    const link = screen.getByRole("link", { name: /open in new tab/i });
    expect(link.getAttribute("href")).toBe(streamValue.url);
    expect(link.getAttribute("target")).toBe("_blank");
  });

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
