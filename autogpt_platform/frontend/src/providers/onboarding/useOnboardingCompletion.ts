import { useState } from "react";
import type { OnboardingCompletion } from "./helpers";

interface Args {
  isLoggedIn: boolean;
  userID: string | null;
}

// The answer is tagged with the account it was read for, so it never stands
// in for another account signed in later on this tab.
export function useOnboardingCompletion({ isLoggedIn, userID }: Args) {
  const [completion, setCompletion] = useState<OnboardingCompletion | null>(
    null,
  );
  const isOnboardingComplete =
    isLoggedIn && completion?.userID === userID ? completion.isCompleted : null;
  return { isOnboardingComplete, setCompletion };
}
