import { PLAN_REQUIRED_MESSAGE } from "../../chat/chatErrors";
import {
  absoluteShareUrl,
  CREATE_FAILED,
  shareErrorText,
  shareUrlForToken,
} from "../helpers";

const WEB = "https://platform.example.com";

describe("absoluteShareUrl", () => {
  it("keeps an absolute URL as sent", () => {
    expect(absoluteShareUrl("https://agpt.co/share/chat/abc", WEB)).toBe(
      "https://agpt.co/share/chat/abc",
    );
  });

  it("prefixes the web's origin to a relative URL", () => {
    expect(absoluteShareUrl("/share/chat/abc", WEB)).toBe(`${WEB}/share/chat/abc`);
    expect(absoluteShareUrl("share/chat/abc", WEB)).toBe(`${WEB}/share/chat/abc`);
  });
});

describe("shareUrlForToken", () => {
  it("builds the viewer path the web uses", () => {
    expect(shareUrlForToken("abc", WEB)).toBe(`${WEB}/share/chat/abc`);
  });
});

describe("shareErrorText", () => {
  it("names the plan for a 402", () => {
    expect(shareErrorText({ status: 402 }, CREATE_FAILED)).toBe(PLAN_REQUIRED_MESSAGE);
  });

  it("falls back for anything else", () => {
    expect(shareErrorText({ status: 500 }, CREATE_FAILED)).toBe(CREATE_FAILED);
    expect(shareErrorText(new Error("offline"), CREATE_FAILED)).toBe(CREATE_FAILED);
  });
});
