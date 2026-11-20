import { closedCopy, deleteErrorCopy } from "../helpers";

describe("deleteErrorCopy", () => {
  it("shows the server's own words for a refusal it phrases", () => {
    expect(deleteErrorCopy({ status: 502, message: "Stripe wouldn't cancel your plan." })).toBe(
      "Stripe wouldn't cancel your plan.",
    );
  });

  it("hides anything else behind one line", () => {
    expect(deleteErrorCopy({ status: 500, message: "KeyError: 'user'" })).toBe(
      "We couldn't delete your account. Try again.",
    );
    expect(deleteErrorCopy(new TypeError("Network request failed"))).toBe(
      "We couldn't delete your account. Try again.",
    );
  });
});

describe("closedCopy", () => {
  it("names the last day to change your mind", () => {
    expect(closedCopy("2026-12-04T12:00:00Z")).toBe(
      "Changed your mind? Sign in on the web before Dec 4 and choose Keep my account.",
    );
  });

  it("falls back to the 14 days when the server gives no date", () => {
    expect(closedCopy(null)).toBe(
      "Changed your mind? Sign in on the web within 14 days and choose Keep my account.",
    );
  });
});
