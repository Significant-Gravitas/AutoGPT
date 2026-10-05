import { AsyncLocalStorage } from "node:async_hooks";

// The sign-up page's `next`, carried from the sign-up action to the repeat
// sign-up email (existing-user-sign-up.ts): Better Auth calls that hook with
// the user alone, so the destination the first sign-up's link got would be
// lost otherwise.
const signUpNext = new AsyncLocalStorage<string | null>();

export function runSignUp<T>(
  next: string | null | undefined,
  signUp: () => Promise<T>,
) {
  return signUpNext.run(next ?? null, signUp);
}

export function getSignUpNext() {
  return signUpNext.getStore() ?? null;
}
