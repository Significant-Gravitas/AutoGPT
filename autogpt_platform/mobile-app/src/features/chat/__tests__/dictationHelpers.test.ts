import {
  appendDictation,
  DICTATION_UNAVAILABLE_MESSAGE,
  elapsedLabel,
  meterLevel,
  NOT_CAUGHT_MESSAGE,
  transcribeFailureMessage,
} from "../dictationHelpers";

describe("appendDictation", () => {
  it("fills an empty field", () => {
    expect(appendDictation("", " hello ")).toBe("hello");
    expect(appendDictation("   ", "hello")).toBe("hello");
  });

  it("adds a space after typed text", () => {
    expect(appendDictation("Draft the email", "to Sam")).toBe("Draft the email to Sam");
  });

  it("adds no second space or break", () => {
    expect(appendDictation("Draft ", "to Sam")).toBe("Draft to Sam");
    expect(appendDictation("Line one\n", "two")).toBe("Line one\ntwo");
  });

  it("keeps the field as it was when nothing was heard", () => {
    expect(appendDictation("Draft", "  ")).toBe("Draft");
  });
});

describe("elapsedLabel", () => {
  it("shows minutes and two-digit seconds", () => {
    expect(elapsedLabel(0)).toBe("0:00");
    expect(elapsedLabel(4_300)).toBe("0:04");
    expect(elapsedLabel(90_000)).toBe("1:30");
    expect(elapsedLabel(120_000)).toBe("2:00");
  });
});

describe("transcribeFailureMessage", () => {
  it("says dictation is unavailable for a 401 or a server failure", () => {
    expect(transcribeFailureMessage(401)).toBe(DICTATION_UNAVAILABLE_MESSAGE);
    expect(transcribeFailureMessage(500)).toBe(DICTATION_UNAVAILABLE_MESSAGE);
    expect(transcribeFailureMessage(503)).toBe(DICTATION_UNAVAILABLE_MESSAGE);
  });

  it("asks to try again for anything else", () => {
    expect(transcribeFailureMessage(400)).toBe(NOT_CAUGHT_MESSAGE);
    expect(transcribeFailureMessage(413)).toBe(NOT_CAUGHT_MESSAGE);
    expect(transcribeFailureMessage(null)).toBe(NOT_CAUGHT_MESSAGE);
  });
});

describe("meterLevel", () => {
  it("maps decibels onto 0 to 1", () => {
    expect(meterLevel(undefined)).toBe(0);
    expect(meterLevel(-160)).toBe(0);
    expect(meterLevel(-30)).toBeCloseTo(0.5);
    expect(meterLevel(0)).toBe(1);
  });
});
