import { beforeEach, describe, expect, it } from "vitest";
import {
  ACCOUNT_CREATED_COOKIE,
  clearAccountCreatedFlag,
  readAccountCreatedFlag,
} from "./account-created-cookie";

describe("account-created flag", () => {
  beforeEach(() => {
    clearAccountCreatedFlag();
  });

  it("reads the signup method without consuming it", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=google; Path=/`;

    expect(readAccountCreatedFlag()).toBe("google");
    // Reading twice still works: the flag survives until the conversion is
    // actually reported.
    expect(readAccountCreatedFlag()).toBe("google");
  });

  it("clears the flag so a reload cannot report twice", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=email; Path=/`;

    clearAccountCreatedFlag();

    expect(readAccountCreatedFlag()).toBeNull();
    expect(document.cookie).not.toContain(`${ACCOUNT_CREATED_COOKIE}=email`);
  });

  it("returns nothing when no account was just created", () => {
    expect(readAccountCreatedFlag()).toBeNull();
  });

  it("discards a value that is not a known signup method", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=weird; Path=/`;

    expect(readAccountCreatedFlag()).toBeNull();
  });
});
