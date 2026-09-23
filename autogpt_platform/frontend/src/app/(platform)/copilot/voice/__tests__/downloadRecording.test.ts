import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  downloadRecording,
  recordingExtension,
  recordingFileName,
} from "../downloadRecording";

describe("recordingExtension", () => {
  it("maps the types MediaRecorder and the VAD actually produce", () => {
    expect(recordingExtension(new Blob([], { type: "audio/webm" }))).toBe(
      "webm",
    );
    expect(recordingExtension(new Blob([], { type: "audio/mp4" }))).toBe("m4a");
    expect(recordingExtension(new Blob([], { type: "audio/wav" }))).toBe("wav");
  });

  it("ignores the codec parameter MediaRecorder appends", () => {
    const blob = new Blob([], { type: "audio/webm;codecs=opus" });

    expect(recordingExtension(blob)).toBe("webm");
  });

  it("falls back rather than producing an extensionless file", () => {
    expect(recordingExtension(new Blob(["x"]))).toBe("webm");
  });
});

describe("recordingFileName", () => {
  it("stamps the name with a time, and no characters a filesystem refuses", () => {
    const name = recordingFileName(
      new Blob([], { type: "audio/wav" }),
      new Date(Date.UTC(2026, 8, 15, 9, 41, 7)),
    );

    expect(name).toBe("voice-recording-2026-09-15-09-41-07.wav");
  });
});

describe("downloadRecording", () => {
  const createObjectURL = vi.fn(() => "blob:recording");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    global.URL.createObjectURL = createObjectURL;
    global.URL.revokeObjectURL = revokeObjectURL;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("clicks a download link for the blob and cleans the URL up after", () => {
    const clicked: HTMLAnchorElement[] = [];
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        clicked.push(this);
      });
    const blob = new Blob(["audio"], { type: "audio/webm" });

    downloadRecording(blob, new Date(Date.UTC(2026, 8, 15, 9, 41, 7)));

    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(clicked).toHaveLength(1);
    expect(clicked[0].getAttribute("download")).toBe(
      "voice-recording-2026-09-15-09-41-07.webm",
    );
    expect(clicked[0].getAttribute("href")).toBe("blob:recording");
    // Removed from the document straight away, so a second failure does not
    // leave a row of invisible links behind.
    expect(document.querySelector("a[download]")).toBeNull();

    expect(revokeObjectURL).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:recording");

    click.mockRestore();
  });
});
