import { afterEach, describe, expect, it, vi } from "vitest";
import { getDatafastAttribution } from "./datafast-attribution";

function setCookie(value: string) {
  vi.spyOn(document, "cookie", "get").mockReturnValue(value);
}

afterEach(() => vi.restoreAllMocks());

describe("getDatafastAttribution", () => {
  it("returns both header values when both cookies are present", () => {
    setCookie("datafast_visitor_id=vis_1; datafast_session_id=ses_1");
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Visitor-Id": "vis_1",
      "X-Datafast-Session-Id": "ses_1",
    });
  });

  it("includes only the cookies that are present", () => {
    setCookie("datafast_visitor_id=vis_1");
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Visitor-Id": "vis_1",
    });
  });

  it("returns an empty object when no DataFast cookies exist", () => {
    setCookie("other=1");
    expect(getDatafastAttribution()).toEqual({});
  });

  it("does not throw on a malformed percent-encoded unrelated cookie", () => {
    setCookie("other=%E0%A4%A; datafast_visitor_id=vis_1");
    expect(() => getDatafastAttribution()).not.toThrow();
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Visitor-Id": "vis_1",
    });
  });

  it("skips a DataFast cookie whose own value is malformed", () => {
    setCookie("datafast_visitor_id=%E0%A4%A; datafast_session_id=ses_1");
    expect(() => getDatafastAttribution()).not.toThrow();
    expect(getDatafastAttribution()).toEqual({
      "X-Datafast-Session-Id": "ses_1",
    });
  });
});
