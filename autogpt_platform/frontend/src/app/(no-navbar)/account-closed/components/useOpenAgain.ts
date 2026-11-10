import { useEffect, useId } from "react";

export function useOpenAgain(isReady: boolean) {
  const headingID = useId();
  const continueID = useId();

  useEffect(
    function focusTheNewHeading() {
      // This card replaces the one whose Keep button had focus.
      document.getElementById(headingID)?.focus();
    },
    [headingID],
  );

  useEffect(
    function focusContinueWhenItAppears() {
      if (isReady) document.getElementById(continueID)?.focus();
    },
    [isReady, continueID],
  );

  return {
    headingID,
    continueID,
    statusText: isReady
      ? "Your account is open everywhere. You can continue."
      : "Opening your account everywhere. This takes up to a minute.",
  };
}
