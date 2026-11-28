import { expect, test } from "./fixtures";

/**
 * Pins an open bug, right before the one check it breaks: from here on the
 * test is expected to fail, and only on that check. CI's report validator
 * (.github/scripts/validate_playwright_json.py) accepts the failure only when
 * every error the test reports is this check, so a cleanup, the journey guard
 * at teardown or any other later failure still fails the run. Anything that
 * must still pass goes above the pin.
 *
 * `bug` names the issue ("#123: what goes wrong"); give the returned message
 * to the check:
 *
 *   const pinned = pinKnownFailure("#123: re-hiring resumes it", "it should stay paused");
 *   expect(state, pinned).toBe("paused");
 */
export function pinKnownFailure(bug: string, check: string) {
  // The validator compares an error's first line, trimmed, with `check`: any
  // other message could never match, and the pin would always be rejected.
  if (!check || check.includes("\n") || check !== check.trim()) {
    throw new Error(
      `pinKnownFailure needs a one-line check message, got ${JSON.stringify(check)}`,
    );
  }
  // A soft check that failed earlier would otherwise end the test "failed" too.
  expect(
    test.info().errors.map((error) => error.message),
    "nothing before the known failure should have failed",
  ).toEqual([]);
  test.info().annotations.push({ type: "known-failure", description: check });
  test.fail(true, bug);
  return check;
}
