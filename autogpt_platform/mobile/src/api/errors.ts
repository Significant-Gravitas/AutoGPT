import { asRecord, asString, nonEmptyString } from "../lib/json";

/** The server answered with a non-success status. Never carries credentials. */
export class HttpError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly detail: unknown;
  readonly retryAfterSeconds: number | null;

  constructor(args: {
    status: number;
    detail: unknown;
    retryAfterSeconds: number | null;
  }) {
    const { message, code } = describeDetail(args.detail);
    super(message ?? `Request failed (${args.status})`);
    this.name = "HttpError";
    this.status = args.status;
    this.code = code;
    this.detail = args.detail;
    this.retryAfterSeconds = args.retryAfterSeconds;
  }
}

/** The request never produced a usable response: offline, DNS, TLS or timeout. */
export class NetworkError extends Error {
  readonly timedOut: boolean;

  constructor(timedOut: boolean) {
    super(
      timedOut
        ? "The server took too long to respond."
        : "Can't reach the server.",
    );
    this.name = "NetworkError";
    this.timedOut = timedOut;
  }
}

/** Credentials are never sent across a redirect, so a redirect is a failure. */
export class RedirectRefusedError extends Error {
  constructor() {
    super(
      "The server redirected the request somewhere else, so it wasn't sent.",
    );
    this.name = "RedirectRefusedError";
  }
}

export class AbortedError extends Error {
  constructor() {
    super("The request was cancelled.");
    this.name = "AbortedError";
  }
}

/** The auth server says this session no longer exists (expired, revoked or signed out). */
export class SessionEndedError extends Error {
  constructor() {
    super("Your session ended. Sign in again.");
    this.name = "SessionEndedError";
  }
}

/** The API refused a freshly issued access token. The stored session is left intact. */
export class SessionRejectedError extends Error {
  constructor() {
    super(
      "This server didn't accept your session. Try again, or sign out and sign back in.",
    );
    this.name = "SessionRejectedError";
  }
}

/** The auth service could not decide right now; the session must be kept. */
export class AuthUnavailableError extends Error {
  readonly retryAfterSeconds: number | null;

  constructor(retryAfterSeconds: number | null) {
    super("Sign-in service is temporarily unavailable. Try again shortly.");
    this.name = "AuthUnavailableError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof AbortedError;
}

export function errorMessage(error: unknown, fallback: string): string {
  if (
    error instanceof HttpError ||
    error instanceof NetworkError ||
    error instanceof RedirectRefusedError ||
    error instanceof SessionEndedError ||
    error instanceof SessionRejectedError ||
    error instanceof AuthUnavailableError
  ) {
    return error.message;
  }
  return fallback;
}

/**
 * FastAPI puts a string or an object in `detail`; Better Auth uses
 * `{code, message}`. Validation arrays become a generic sentence.
 */
function describeDetail(body: unknown): {
  message: string | null;
  code: string | null;
} {
  if (typeof body === "string") {
    return { message: nonEmptyString(body), code: null };
  }
  const record = asRecord(body);
  if (!record) return { message: null, code: null };
  const detail = record.detail ?? record;
  if (typeof detail === "string") {
    return {
      message: nonEmptyString(detail),
      code: detail.match(/^[a-z0-9_]+$/) ? detail : null,
    };
  }
  if (Array.isArray(detail)) {
    return {
      message: "The server couldn't accept that request.",
      code: "validation_error",
    };
  }
  const object = asRecord(detail);
  if (!object) return { message: null, code: null };
  return {
    message: nonEmptyString(object.message),
    code: asString(object.code) ?? asString(object.kind),
  };
}
