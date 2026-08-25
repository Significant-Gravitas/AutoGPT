import { beforeEach, describe, expect, it } from "vitest";
import {
  ACCOUNT_CREATED_COOKIE,
  consumeAccountCreatedFlag,
} from "./account-created-cookie";

describe("consumeAccountCreatedFlag", () => {
  beforeEach(() => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=; Path=/; Max-Age=0`;
  });

  it("returns the signup method once and clears the flag", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=google; Path=/`;

    expect(consumeAccountCreatedFlag()).toBe("google");
    expect(document.cookie).not.toContain(`${ACCOUNT_CREATED_COOKIE}=google`);
    expect(consumeAccountCreatedFlag()).toBeNull();
  });

  it("returns nothing when no account was just created", () => {
    expect(consumeAccountCreatedFlag()).toBeNull();
  });

  it("discards a value that is not a known signup method", () => {
    document.cookie = `${ACCOUNT_CREATED_COOKIE}=weird; Path=/`;

    expect(consumeAccountCreatedFlag()).toBeNull();
    expect(document.cookie).not.toContain(`${ACCOUNT_CREATED_COOKIE}=weird`);
  });
});
