// No imports, so code that only needs to recognise the error (and its tests)
// does not load the auth client.
export class SignedOutError extends Error {
  constructor() {
    super("You are signed out. Sign in again to continue.");
    this.name = "SignedOutError";
  }
}

/**
 * The account changed while a request was on its way. Not a dead session:
 * whoever is signed in now stays signed in, and the request is not sent for them.
 */
export class SessionChangedError extends Error {
  constructor() {
    super("You signed in again while this was loading. Try again.");
    this.name = "SessionChangedError";
  }
}
