import { NO_APP_SESSION_CODE } from "@/lib/auth/sessionHelpers";

import {
  canSubmitSignup,
  passwordProblem,
  resendLabel,
  signupErrorCopy,
  verifyErrorCopy,
} from "../signupHelpers";

const OFFLINE = "Can't reach AutoGPT. Check your connection and try again.";

describe("canSubmitSignup", () => {
  it("needs an email and a password", () => {
    expect(canSubmitSignup("  ", "x")).toBe(false);
    expect(canSubmitSignup("ada@northwind.co", "")).toBe(false);
    expect(canSubmitSignup(" ada@northwind.co ", "x")).toBe(true);
  });
});

describe("passwordProblem", () => {
  it("asks for 12 characters", () => {
    expect(passwordProblem("elevenchars")).toBe(
      "Your password needs at least 12 characters.",
    );
    expect(passwordProblem("twelve-chars")).toBeNull();
  });
});

describe("signupErrorCopy", () => {
  it("says the server can't be reached when there was no answer", () => {
    expect(signupErrorCopy(null)).toBe(OFFLINE);
  });

  it.each(["USER_ALREADY_EXISTS", "USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL"])(
    "points an existing account at sign-in: %s",
    (code) => {
      expect(signupErrorCopy({ status: 422, code })).toBe(
        "An account with this email already exists. Sign in instead.",
      );
    },
  );

  it("repeats the password rule when the server enforces it", () => {
    expect(signupErrorCopy({ status: 400, code: "PASSWORD_TOO_SHORT" })).toBe(
      "Your password needs at least 12 characters.",
    );
  });

  it("names a server that can't sign the app in", () => {
    expect(signupErrorCopy({ status: 200, code: NO_APP_SESSION_CODE })).toBe(
      "This server can't sign the app in yet. Update it and try again.",
    );
  });

  it("asks for a pause when rate limited", () => {
    expect(signupErrorCopy({ status: 429 })).toBe(
      "Too many tries. Wait a minute, then try again.",
    );
  });

  it("shows the server's own refusal, such as a closed sign-up list", () => {
    const message =
      "This email address is not allowed to register. Please contact support for assistance.";
    expect(signupErrorCopy({ status: 403, message })).toBe(message);
    expect(signupErrorCopy({ status: 500 })).toBe(
      "You can't sign up right now. Try again later.",
    );
  });
});

describe("verifyErrorCopy", () => {
  it("says the link has not been opened yet", () => {
    expect(verifyErrorCopy({ status: 403, code: "EMAIL_NOT_VERIFIED" })).toBe(
      "Not verified yet. Open the link in the email, then try again.",
    );
  });

  it("falls back like sign-in", () => {
    expect(verifyErrorCopy(null)).toBe(OFFLINE);
    expect(verifyErrorCopy({ status: 500 })).toBe(
      "You can't sign in right now. Try again later.",
    );
  });
});

describe("resendLabel", () => {
  it("counts down, then offers the resend", () => {
    expect(resendLabel(42)).toBe("Resend email in 42s");
    expect(resendLabel(0)).toBe("Resend email");
  });
});
