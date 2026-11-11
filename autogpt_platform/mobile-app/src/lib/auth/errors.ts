// No imports, so code that only needs to recognise the error (and its tests)
// does not load the auth client.
export class SignedOutError extends Error {
  constructor() {
    super("You are signed out. Sign in again to continue.");
    this.name = "SignedOutError";
  }
}
