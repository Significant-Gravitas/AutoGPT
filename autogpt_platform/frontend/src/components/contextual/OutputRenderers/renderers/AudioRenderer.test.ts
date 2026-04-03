import { describe, expect, it } from "vitest";
import { audioRenderer } from "./AudioRenderer";

describe("AudioRenderer canRender", () => {
  it("detects audio file URLs by extension", () => {
    expect(audioRenderer.canRender("https://example.com/song.mp3")).toBe(true);
    expect(audioRenderer.canRender("https://example.com/track.wav")).toBe(true);
    expect(audioRenderer.canRender("https://example.com/audio.ogg")).toBe(true);
    expect(audioRenderer.canRender("https://example.com/music.flac")).toBe(
      true,
    );
    expect(audioRenderer.canRender("https://example.com/podcast.m4a")).toBe(
      true,
    );
    expect(audioRenderer.canRender("https://example.com/voice.aac")).toBe(true);
  });

  it("defers .webm to VideoRenderer (higher priority)", () => {
    expect(audioRenderer.canRender("https://example.com/sound.webm")).toBe(
      false,
    );
  });

  it("detects audio data URIs", () => {
    expect(audioRenderer.canRender("data:audio/mpeg;base64,abc")).toBe(true);
    expect(audioRenderer.canRender("data:audio/wav;base64,abc")).toBe(true);
  });

  it("detects via metadata type", () => {
    expect(audioRenderer.canRender("anything", { type: "audio" })).toBe(true);
  });

  it("detects via metadata mimeType", () => {
    expect(
      audioRenderer.canRender("anything", { mimeType: "audio/mpeg" }),
    ).toBe(true);
    expect(audioRenderer.canRender("anything", { mimeType: "audio/wav" })).toBe(
      true,
    );
  });

  it("rejects non-audio URLs", () => {
    expect(audioRenderer.canRender("https://example.com/video.mp4")).toBe(
      false,
    );
    expect(audioRenderer.canRender("https://example.com/image.png")).toBe(
      false,
    );
    expect(audioRenderer.canRender("https://example.com/page")).toBe(false);
    expect(audioRenderer.canRender("just some text")).toBe(false);
  });

  it("rejects non-string values", () => {
    expect(audioRenderer.canRender(123)).toBe(false);
    expect(audioRenderer.canRender(null)).toBe(false);
    expect(audioRenderer.canRender({ url: "test.mp3" })).toBe(false);
  });
});
