import { describe, expect, it } from "vitest";
import {
  getYouTubeEmbedUrl,
  getYouTubeVideoId,
  getYouTubeWatchUrl,
  isYouTubeImageUrl,
  isYouTubeUrl,
} from "../helpers";

const ID = "dQw4w9WgXcQ";

describe("getYouTubeVideoId", () => {
  it.each([
    `https://www.youtube.com/watch?v=${ID}`,
    `https://youtube.com/watch?v=${ID}&t=30s`,
    `https://www.youtube.com/watch?feature=shared&v=${ID}`,
    `https://m.youtube.com/watch?v=${ID}`,
    `https://music.youtube.com/watch?v=${ID}&list=abc`,
    `https://youtu.be/${ID}`,
    `https://youtu.be/${ID}?si=abc123`,
    `https://www.youtube.com/shorts/${ID}`,
    `https://youtube.com/shorts/${ID}?feature=share`,
    `https://www.youtube.com/live/${ID}`,
    `https://www.youtube.com/embed/${ID}`,
    `https://www.youtube.com/v/${ID}`,
    `https://www.youtube-nocookie.com/embed/${ID}`,
    `www.youtube.com/watch?v=${ID}`,
    `youtu.be/${ID}`,
  ])("finds the video in %s", (url) => {
    expect(getYouTubeVideoId(url)).toBe(ID);
  });

  it.each([
    "https://www.youtube.com/@somechannel",
    "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/shorts/",
    "https://www.youtube.com/embed/videoseries?list=PL123",
    "https://www.youtube.com/embed/live_stream?channel=UC123",
    `https://youtube.com.example.com/watch?v=${ID}`,
    `https://example.com/watch?v=${ID}`,
    `javascript:alert(1)//youtu.be/${ID}`,
  ])("finds no video in %s", (url) => {
    expect(getYouTubeVideoId(url)).toBeNull();
  });
});

describe("isYouTubeUrl", () => {
  it("accepts any YouTube page", () => {
    expect(isYouTubeUrl("https://www.youtube.com/@somechannel")).toBe(true);
    expect(isYouTubeUrl("https://m.youtube.com/playlist?list=PL1")).toBe(true);
  });

  it("rejects other hosts that only look like YouTube", () => {
    expect(isYouTubeUrl("https://youtube.com.example.com/x")).toBe(false);
    expect(isYouTubeUrl("https://notyoutube.com/watch?v=x")).toBe(false);
    expect(isYouTubeUrl("https://example.com/video.mp4")).toBe(false);
  });

  it("leaves YouTube image hosts to render as images", () => {
    expect(isYouTubeUrl(`https://img.youtube.com/vi/${ID}/0.jpg`)).toBe(false);
    expect(isYouTubeUrl(`https://i.ytimg.com/vi/${ID}/hqdefault.jpg`)).toBe(
      false,
    );
  });
});

describe("getYouTubeWatchUrl", () => {
  it("links a video to its watch page", () => {
    expect(getYouTubeWatchUrl(`https://youtube.com/shorts/${ID}`)).toBe(
      `https://www.youtube.com/watch?v=${ID}`,
    );
  });

  it("keeps any other YouTube link as given", () => {
    expect(getYouTubeWatchUrl("youtube.com/@somechannel")).toBe(
      "https://youtube.com/@somechannel",
    );
  });

  it("gives nothing for a link outside YouTube", () => {
    expect(getYouTubeWatchUrl("https://example.com/x")).toBeNull();
  });
});

describe("start time", () => {
  it.each([
    [`https://youtu.be/${ID}?t=90`, 90],
    [`https://www.youtube.com/watch?v=${ID}&t=1m30s`, 90],
    [`https://www.youtube.com/watch?v=${ID}&t=1h2m3s`, 3723],
    [`https://www.youtube.com/embed/${ID}?start=45`, 45],
  ])("keeps the start of %s", (url, seconds) => {
    expect(getYouTubeEmbedUrl(url)).toBe(
      `https://www.youtube-nocookie.com/embed/${ID}?start=${seconds}`,
    );
    expect(getYouTubeWatchUrl(url)).toBe(
      `https://www.youtube.com/watch?v=${ID}&t=${seconds}s`,
    );
  });

  it("drops a start it can't read and every other parameter", () => {
    const url = `https://youtu.be/${ID}?t=soon&si=tracking`;
    expect(getYouTubeEmbedUrl(url)).toBe(
      `https://www.youtube-nocookie.com/embed/${ID}`,
    );
    expect(getYouTubeWatchUrl(url)).toBe(
      `https://www.youtube.com/watch?v=${ID}`,
    );
  });
});

describe("isYouTubeImageUrl", () => {
  it("recognises YouTube image hosts", () => {
    expect(isYouTubeImageUrl(`https://img.youtube.com/vi/${ID}/0.jpg`)).toBe(
      true,
    );
    expect(isYouTubeImageUrl(`https://i.ytimg.com/vi/${ID}/0.jpg`)).toBe(true);
    expect(isYouTubeImageUrl(`https://i3.ytimg.com/vi/${ID}/0.jpg`)).toBe(true);
  });

  it("ignores YouTube pages and other hosts", () => {
    expect(isYouTubeImageUrl(`https://www.youtube.com/watch?v=${ID}`)).toBe(
      false,
    );
    expect(isYouTubeImageUrl("https://ytimg.com.example.com/a.jpg")).toBe(
      false,
    );
    expect(isYouTubeImageUrl("https://cdn.test/one.png")).toBe(false);
  });
});

describe("hosts written with a trailing dot", () => {
  it.each([
    [`https://www.youtube.com./watch?v=${ID}`, ID],
    [`https://YOUTUBE.COM./shorts/${ID}`, ID],
    [`https://youtu.be./${ID}`, ID],
  ])("still finds the video in %s", (url, id) => {
    expect(isYouTubeUrl(url)).toBe(true);
    expect(getYouTubeVideoId(url)).toBe(id);
  });

  it.each([
    `https://img.youtube.com./vi/${ID}/0.jpg`,
    `https://i.ytimg.com./vi/${ID}/mqdefault.jpg`,
  ])("still sees %s as a YouTube image", (url) => {
    expect(isYouTubeImageUrl(url)).toBe(true);
  });
});
