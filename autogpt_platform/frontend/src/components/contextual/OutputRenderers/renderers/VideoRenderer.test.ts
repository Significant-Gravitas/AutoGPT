import { describe, expect, it } from "vitest";
import { videoRenderer } from "./VideoRenderer";

describe("VideoRenderer canRender", () => {
  it("detects direct video file URLs", () => {
    expect(videoRenderer.canRender("https://example.com/video.mp4")).toBe(true);
    expect(videoRenderer.canRender("https://example.com/video.webm")).toBe(
      true,
    );
    expect(videoRenderer.canRender("https://example.com/video.mov")).toBe(true);
  });

  it("detects data URIs", () => {
    expect(videoRenderer.canRender("data:video/mp4;base64,abc")).toBe(true);
  });

  it("detects YouTube watch URLs", () => {
    expect(
      videoRenderer.canRender("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
    ).toBe(true);
    expect(
      videoRenderer.canRender("https://youtube.com/watch?v=dQw4w9WgXcQ"),
    ).toBe(true);
  });

  it("detects YouTube short URLs", () => {
    expect(videoRenderer.canRender("https://youtu.be/dQw4w9WgXcQ")).toBe(true);
  });

  it("detects YouTube embed URLs", () => {
    expect(
      videoRenderer.canRender("https://www.youtube.com/embed/dQw4w9WgXcQ"),
    ).toBe(true);
  });

  it("detects Vimeo URLs", () => {
    expect(videoRenderer.canRender("https://vimeo.com/123456789")).toBe(true);
    expect(videoRenderer.canRender("https://www.vimeo.com/123456789")).toBe(
      true,
    );
  });

  it("rejects non-video URLs", () => {
    expect(videoRenderer.canRender("https://example.com/page")).toBe(false);
    expect(videoRenderer.canRender("https://example.com/image.png")).toBe(
      false,
    );
    expect(videoRenderer.canRender("just some text")).toBe(false);
  });

  it("handles metadata type override", () => {
    expect(videoRenderer.canRender("anything", { type: "video" })).toBe(true);
  });
});
