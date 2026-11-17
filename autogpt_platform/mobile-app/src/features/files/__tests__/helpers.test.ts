import {
  capText,
  decodeUtf8,
  isPreviewed,
  TEXT_DISPLAY_LIMIT,
  trimPartialUtf8,
} from "../helpers";

function trimmed(bytes: number[]) {
  return Array.from(trimPartialUtf8(new Uint8Array(bytes)));
}

// "\u00e9" is C3 A9, "\u20ac" is E2 82 AC, "\ud83d\ude00" is F0 9F 98 80.
const MULTIBYTE = [0xc3, 0xa9, 0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80];

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
});

describe("trimPartialUtf8", () => {
  it("leaves ASCII alone", () => {
    expect(trimmed([0x68, 0x69])).toEqual([0x68, 0x69]);
  });

  it("drops a two-byte character cut after its lead byte", () => {
    expect(trimmed([0x63, 0x61, 0x66, 0xc3])).toEqual([0x63, 0x61, 0x66]);
  });

  it("drops a four-byte emoji cut part way", () => {
    expect(trimmed([0x61, 0xf0, 0x9f, 0x98])).toEqual([0x61]);
  });

  it("leaves complete multibyte characters alone", () => {
    expect(trimmed(MULTIBYTE)).toEqual(MULTIBYTE);
  });
});

describe("decodeUtf8", () => {
  it("decodes ASCII", () => {
    expect(decodeUtf8(new Uint8Array([0x68, 0x69]))).toBe("hi");
  });

  it("decodes two-, three- and four-byte characters", () => {
    expect(decodeUtf8(new Uint8Array(MULTIBYTE))).toBe(
      "\u00e9\u20ac\ud83d\ude00",
    );
  });

  it("turns a bad byte into U+FFFD and keeps going", () => {
    expect(decodeUtf8(new Uint8Array([0x61, 0xff, 0x62]))).toBe("a\ufffdb");
  });

  it("turns a cut character into one U+FFFD", () => {
    expect(decodeUtf8(new Uint8Array([0x61, 0xf0, 0x9f, 0x98]))).toBe(
      "a\ufffd",
    );
  });
});
