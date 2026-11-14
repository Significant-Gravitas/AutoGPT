import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  isAccountClosedResponse,
  showAccountClosedScreen,
} from "../closed-account";

describe("showAccountClosedScreen", () => {
  let replace: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    replace = vi
      .spyOn(window.location, "replace")
      .mockImplementation(() => undefined);
  });

  afterEach(() => {
    replace.mockRestore();
    window.history.replaceState({}, "", "/");
  });

  test("sends a refused tab to the keep-account page", () => {
    window.history.replaceState({}, "", "/copilot");

    showAccountClosedScreen();

    expect(replace).toHaveBeenCalledWith("/account-closed");
  });

  test.each(["/account-closed", "/logout"])(
    "stays put on %s: already there, or signing out after deleting",
    (path) => {
      window.history.replaceState({}, "", path);

      showAccountClosedScreen();

      expect(replace).not.toHaveBeenCalled();
    },
  );
});

describe("isAccountClosedResponse", () => {
  test("only the backend's 401 with the closed detail", () => {
    expect(
      isAccountClosedResponse(401, { detail: "This account is closed." }),
    ).toBe(true);
    expect(
      isAccountClosedResponse(403, { detail: "This account is closed." }),
    ).toBe(false);
    expect(isAccountClosedResponse(401, { detail: "Not authenticated" })).toBe(
      false,
    );
    expect(isAccountClosedResponse(401, null)).toBe(false);
  });
});
