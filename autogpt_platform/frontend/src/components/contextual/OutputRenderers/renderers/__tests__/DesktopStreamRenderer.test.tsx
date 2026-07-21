import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { desktopStreamRenderer } from "../DesktopStreamRenderer";

const streamValue = {
  kind: "desktop_stream",
  url: "https://6080-sandbox.e2b.app/vnc.html?autoconnect=true",
  provider: "e2b",
  sandbox_id: "sbx-123",
  requires_auth: false,
};

describe("DesktopStreamRenderer", () => {
  afterEach(() => {
    cleanup();
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

  it("copies the stream URL", () => {
    const copy = desktopStreamRenderer.getCopyContent(streamValue);
    expect(copy?.data).toBe(streamValue.url);
  });

  it("has no download content", () => {
    expect(desktopStreamRenderer.getDownloadContent(streamValue)).toBeNull();
  });
});
