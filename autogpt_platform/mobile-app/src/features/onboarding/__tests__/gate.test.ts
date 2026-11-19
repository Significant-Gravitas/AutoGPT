import { gateState, onboardedKey } from "../gate";

describe("gateState", () => {
  const asked = { isRemembered: false, isCompleted: undefined } as const;

  it("skips the question for an account this phone already knows", () => {
    expect(
      gateState({ isRemembered: true, status: "pending", isCompleted: undefined }),
    ).toBe("done");
  });

  it("waits while the server is asked", () => {
    expect(gateState({ ...asked, status: "pending" })).toBe("loading");
  });

  it("shows onboarding to an account that has not finished it", () => {
    expect(
      gateState({ isRemembered: false, status: "success", isCompleted: false }),
    ).toBe("needed");
  });

  it("lets a finished account in", () => {
    expect(
      gateState({ isRemembered: false, status: "success", isCompleted: true }),
    ).toBe("done");
  });

  it("lets the person in when the server can't be asked", () => {
    expect(gateState({ ...asked, status: "error" })).toBe("done");
  });
});

describe("onboardedKey", () => {
  it("is one answer per account and per server", () => {
    const key = onboardedKey("https://platform.agpt.co", "user-1");
    expect(key).not.toBe(onboardedKey("https://platform.agpt.co", "user-2"));
    expect(key).not.toBe(onboardedKey("http://localhost:3000", "user-1"));
    expect(key).toMatch(/^[A-Za-z0-9._-]+$/);
  });
});
