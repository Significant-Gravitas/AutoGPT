import { capText, isPreviewed, TEXT_DISPLAY_LIMIT } from "../helpers";

describe("isPreviewed", () => {
  it("downloads a file up to the display limit whole", () => {
    expect(isPreviewed({ size_bytes: TEXT_DISPLAY_LIMIT })).toBe(false);
  });

  it("previews a file over the display limit", () => {
    expect(isPreviewed({ size_bytes: TEXT_DISPLAY_LIMIT + 1 })).toBe(true);
  });
});

describe("capText", () => {
  it("shows a short downloaded file in full", () => {
    expect(capText({ text: "hello", isPreview: false })).toEqual({
      text: "hello",
      cappedNotice: null,
    });
  });

  it("cuts long downloaded text at 200 KB", () => {
    const result = capText({
      text: "a".repeat(TEXT_DISPLAY_LIMIT + 10),
      isPreview: false,
    });

    expect(result.text).toHaveLength(TEXT_DISPLAY_LIMIT);
    expect(result.cappedNotice).toBe("Showing the first 200 KB.");
  });

  it("marks a preview as the first 128 KB", () => {
    expect(capText({ text: "first part", isPreview: true })).toEqual({
      text: "first part",
      cappedNotice: "Showing the first 128 KB.",
    });
  });

  it("drops the half character a preview's byte cut leaves at the end", () => {
    expect(capText({ text: "caf�", isPreview: true }).text).toBe("caf");
  });
});
