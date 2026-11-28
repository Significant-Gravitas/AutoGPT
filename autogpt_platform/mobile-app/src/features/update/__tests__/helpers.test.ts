import { isBelowMinimum, storeURL, updateGate } from "../helpers";

describe("isBelowMinimum", () => {
  it("compares part by part, as numbers", () => {
    expect(isBelowMinimum("0.1.0", "0.2.0")).toBe(true);
    expect(isBelowMinimum("0.9.0", "0.10.0")).toBe(true);
    expect(isBelowMinimum("1.0.0", "0.10.0")).toBe(false);
    expect(isBelowMinimum("0.2.0", "0.2.0")).toBe(false);
    expect(isBelowMinimum("0.2.1", "0.2.0")).toBe(false);
  });

  it("fills missing parts with zeros and ignores what follows a hyphen or plus", () => {
    expect(isBelowMinimum("1.2", "1.2.0")).toBe(false);
    expect(isBelowMinimum("1", "1.0.1")).toBe(true);
    expect(isBelowMinimum("1.2.0-rc.1", "1.2.0")).toBe(false);
    expect(isBelowMinimum("1.2.0+45", "1.2.1")).toBe(true);
  });

  it("lets the app in when either version is missing or unreadable", () => {
    expect(isBelowMinimum(null, "1.0.0")).toBe(false);
    expect(isBelowMinimum("1.0.0", null)).toBe(false);
    expect(isBelowMinimum("1.0.0", "latest")).toBe(false);
    expect(isBelowMinimum("dev", "1.0.0")).toBe(false);
  });
});

describe("updateGate", () => {
  it("waits for the check, then asks for an update only when the app is too old", () => {
    expect(updateGate("pending", null, "0.1.0")).toBe("checking");
    expect(updateGate("success", "0.2.0", "0.1.0")).toBe("required");
    expect(updateGate("success", null, "0.1.0")).toBe("ok");
  });

  it("lets the app in when the server can't be asked", () => {
    expect(updateGate("error", null, "0.1.0")).toBe("ok");
  });
});

describe("storeURL", () => {
  it("opens the Play Store on Android and the App Store once its ID is known", () => {
    expect(storeURL("android", "co.agpt.autogpt.app", null)).toBe("market://details?id=co.agpt.autogpt.app");
    expect(storeURL("ios", null, "123456")).toBe("itms-apps://apps.apple.com/app/id123456");
    expect(storeURL("ios", null, null)).toBeNull();
  });
});
