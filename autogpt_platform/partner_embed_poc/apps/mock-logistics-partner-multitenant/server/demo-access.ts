import { createHmac, timingSafeEqual } from "node:crypto";

const COOKIE_PAYLOAD = "autogpt-partner-demo-access-v1";
export const DEMO_ACCESS_TTL_SECONDS = 8 * 60 * 60;

interface DemoAccessGateOptions {
  now?: () => number;
  required?: boolean;
}

export interface DemoAccessGate {
  enabled: boolean;
  acceptsCode(code: string): boolean;
  acceptsCookie(cookie: string | undefined): boolean;
  cookieValue(): string;
}

export function createDemoAccessGate(
  accessCode: string | undefined,
  options: DemoAccessGateOptions = {},
): DemoAccessGate {
  const code = accessCode ?? "";
  const now = options.now ?? Date.now;
  if (options.required && !code) {
    throw new Error("DEMO_ACCESS_CODE is required in public demo mode");
  }
  if (code && code.length < 16) {
    throw new Error("DEMO_ACCESS_CODE must contain at least 16 characters");
  }
  const expectedCode = digest(code, "code");

  return {
    enabled: Boolean(code),
    acceptsCode(candidate) {
      return (
        Boolean(code) && safeEqual(digest(candidate, "code"), expectedCode)
      );
    },
    acceptsCookie(candidate) {
      if (!code || !candidate) return false;
      const [expiresAtText, signature, extra] = candidate.split(".");
      if (
        extra ||
        !expiresAtText ||
        !signature ||
        !/^\d+$/.test(expiresAtText)
      ) {
        return false;
      }
      const expiresAt = Number(expiresAtText);
      if (!Number.isSafeInteger(expiresAt)) return false;
      if (expiresAt <= Math.floor(now() / 1000)) return false;
      return safeEqual(
        signature,
        digest(code, `${COOKIE_PAYLOAD}.${expiresAtText}`),
      );
    },
    cookieValue() {
      const expiresAt = Math.floor(now() / 1000) + DEMO_ACCESS_TTL_SECONDS;
      return `${expiresAt}.${digest(code, `${COOKIE_PAYLOAD}.${expiresAt}`)}`;
    },
  };
}

export function createDemoAccessRateLimiter(
  maxAttempts = 12,
  windowMs = 15 * 60 * 1000,
  now: () => number = Date.now,
  maxClients = 1024,
) {
  const attempts = new Map<string, { count: number; resetsAt: number }>();

  return {
    consume(key: string) {
      const currentTime = now();
      for (const [trackedKey, attempt] of attempts) {
        if (attempt.resetsAt <= currentTime) attempts.delete(trackedKey);
      }
      const current = attempts.get(key);
      if (!current) {
        if (attempts.size >= maxClients) {
          return {
            allowed: false,
            retryAfterSeconds: Math.max(1, Math.ceil(windowMs / 1000)),
          };
        }
        attempts.set(key, { count: 1, resetsAt: currentTime + windowMs });
        return { allowed: true, retryAfterSeconds: 0 };
      }
      if (current.count >= maxAttempts) {
        return {
          allowed: false,
          retryAfterSeconds: Math.max(
            1,
            Math.ceil((current.resetsAt - currentTime) / 1000),
          ),
        };
      }
      current.count += 1;
      return { allowed: true, retryAfterSeconds: 0 };
    },
    reset(key: string) {
      attempts.delete(key);
    },
  };
}

function digest(secret: string, value: string) {
  return createHmac("sha256", secret || "disabled-demo-access")
    .update(value)
    .digest("base64url");
}

function safeEqual(received: string, expected: string) {
  const receivedBuffer = Buffer.from(received);
  const expectedBuffer = Buffer.from(expected);
  return (
    receivedBuffer.length === expectedBuffer.length &&
    timingSafeEqual(receivedBuffer, expectedBuffer)
  );
}
