import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DesktopStreamPreview } from "../DesktopStreamRenderer";

const streamValue = {
  kind: "desktop_stream" as const,
  url: "https://6080-sandbox.e2b.app/vnc.html?autoconnect=true",
  provider: "e2b",
  sandbox_id: "sbx-123",
  requires_auth: false,
};

describe("DesktopStreamRenderer", () => {
  const browserErrors: unknown[] = [];

  function recordBrowserError(event: ErrorEvent) {
    browserErrors.push(event.error ?? event.message);
  }

  beforeEach(() => {
    browserErrors.length = 0;
    window.addEventListener("error", recordBrowserError);
  });

  const fullscreenEnabledDescriptor = Object.getOwnPropertyDescriptor(
    document,
    "fullscreenEnabled",
  );
  const requestFullscreenDescriptor = Object.getOwnPropertyDescriptor(
    HTMLIFrameElement.prototype,
    "requestFullscreen",
  );

  afterEach(() => {
    cleanup();
    window.removeEventListener("error", recordBrowserError);
    vi.restoreAllMocks();
    for (const [target, key, descriptor] of [
      [document, "fullscreenEnabled", fullscreenEnabledDescriptor],
      [
        HTMLIFrameElement.prototype,
        "requestFullscreen",
        requestFullscreenDescriptor,
      ],
    ] as const) {
      if (descriptor) Object.defineProperty(target, key, descriptor);
      else Reflect.deleteProperty(target, key);
    }
    expect(browserErrors).toEqual([]);
  });

  function mockFullscreenEnabled(enabled: boolean | undefined) {
    Object.defineProperty(document, "fullscreenEnabled", {
      configurable: true,
      get: () => enabled,
    });
  }

  function mockRequestFullscreen(impl: () => Promise<void> | void) {
    const spy = vi.fn();
    function requestFullscreen(this: HTMLIFrameElement) {
      spy.call(this);
      return impl();
    }
    Object.defineProperty(HTMLIFrameElement.prototype, "requestFullscreen", {
      configurable: true,
      writable: true,
      value: requestFullscreen,
    });
    return spy;
  }

  it.each([false, undefined])(
    "hides fullscreen when support is %s",
    (enabled) => {
      mockFullscreenEnabled(enabled);
      render(<DesktopStreamPreview value={streamValue} />);
      expect(screen.queryByRole("button", { name: /fullscreen/i })).toBeNull();
      expect(
        screen.getByRole("link", { name: /open in new tab/i }),
      ).toBeDefined();
    },
  );

  it.each([
    { name: "a promise", impl: () => Promise.resolve() },
    { name: "void", impl: () => undefined },
  ])(
    "requests fullscreen on the frame when the method returns $name",
    async ({ impl }) => {
      const user = userEvent.setup();
      mockFullscreenEnabled(true);
      const requestFullscreen = mockRequestFullscreen(impl);
      render(<DesktopStreamPreview value={streamValue} />);
      const button = await screen.findByRole("button", { name: /fullscreen/i });
      await user.click(button);
      expect(requestFullscreen).toHaveBeenCalledOnce();
      expect(requestFullscreen.mock.instances[0]).toBe(
        screen.getByTitle("Interactive desktop (sbx-123)"),
      );
    },
  );

  it.each([undefined, null, 42, {}])(
    "keeps the stream usable when requestFullscreen is %s",
    async (method) => {
      const user = userEvent.setup();
      mockFullscreenEnabled(true);
      Object.defineProperty(HTMLIFrameElement.prototype, "requestFullscreen", {
        configurable: true,
        writable: true,
        value: method,
      });
      render(<DesktopStreamPreview value={streamValue} />);
      const button = await screen.findByRole("button", { name: /fullscreen/i });
      await user.click(button);
      expect(screen.getByTitle("Interactive desktop (sbx-123)")).toBeDefined();
      expect(
        screen
          .getByRole("link", { name: /open in new tab/i })
          .getAttribute("href"),
      ).toBe(streamValue.url);
    },
  );

  it.each([
    { name: "rejects", impl: () => Promise.reject(new Error("denied")) },
    {
      name: "throws",
      impl: () => {
        throw new Error("denied");
      },
    },
  ])(
    "keeps the stream usable when a fullscreen request $name",
    async ({ impl }) => {
      const user = userEvent.setup();
      mockFullscreenEnabled(true);
      const requestFullscreen = mockRequestFullscreen(impl);
      render(<DesktopStreamPreview value={streamValue} />);
      const button = await screen.findByRole("button", { name: /fullscreen/i });
      await user.click(button);
      expect(requestFullscreen).toHaveBeenCalledOnce();
      expect(screen.getByText("Interactive Desktop")).toBeDefined();
      expect(screen.getByTitle("Interactive desktop (sbx-123)")).toBeDefined();
      expect(
        screen.getByRole("link", { name: /open in new tab/i }),
      ).toBeDefined();
    },
  );

  it("handles a fullscreen rejection after the click handler returns", async () => {
    const user = userEvent.setup();
    mockFullscreenEnabled(true);
    const pending = Promise.withResolvers<void>();
    const requestFullscreen = mockRequestFullscreen(() => pending.promise);
    render(<DesktopStreamPreview value={streamValue} />);
    const frame = screen.getByTitle("Interactive desktop (sbx-123)");

    await user.click(screen.getByRole("button", { name: /fullscreen/i }));
    expect(requestFullscreen).toHaveBeenCalledOnce();
    await act(async () => pending.reject(new Error("denied later")));

    expect(screen.getByTitle("Interactive desktop (sbx-123)")).toBe(frame);
    expect(
      screen.getByRole("link", { name: /open in new tab/i }),
    ).toBeDefined();
  });

  it("hydrates server markup before showing the supported fullscreen control", async () => {
    mockFullscreenEnabled(true);
    const preview = <DesktopStreamPreview value={streamValue} />;
    const markup = renderToString(preview);
    expect(markup).not.toContain('aria-label="Fullscreen"');
    const container = document.createElement("div");
    container.innerHTML = markup;
    document.body.appendChild(container);

    render(preview, { container, hydrate: true });

    expect(
      await screen.findByRole("button", { name: /fullscreen/i }),
    ).toBeDefined();
  });

  it("shows a notice instead of the frame to a viewer who is not the owner", () => {
    mockFullscreenEnabled(true);
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
    expect(screen.queryByRole("button", { name: /fullscreen/i })).toBeNull();
  });
});
