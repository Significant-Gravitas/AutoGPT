import type {
  BrowserContext,
  ConsoleMessage,
  Page,
  Response,
} from "@playwright/test";

/**
 * A response a journey provokes on purpose, e.g. the 401 a signed-out
 * session gets. `url` matches as a substring, or as a RegExp.
 */
export interface ExpectedResponse {
  status: number;
  url: string | RegExp;
  method?: string;
  /** A header the response must carry, to exempt only responses the journey
   * fakes itself and not a real failure on the same URL. */
  header?: { name: string; value: string };
}

interface KnownNoise {
  reason: string;
  kind: "response" | "console" | "pageerror";
  status?: number;
  url?: RegExp;
  message?: RegExp;
}

// Reviewed, app-wide noise that is NOT a bug. Every entry needs a reason a
// reviewer can check. A real defect never goes here: fix it, or pin the
// journey to its issue with pinKnownFailure() (known-failure.ts).
const KNOWN_BENIGN: KnownNoise[] = [
  {
    kind: "console",
    message: /^Failed to load resource: /,
    reason:
      "Chrome's echo of a failed request. Failed API calls are reported " +
      "by the response check, with their URL and body.",
  },
];

const BODY_SNIPPET_LENGTH = 300;

/**
 * Watches every page in a journey's browser context and collects what a user
 * would never want to see: a 4xx/5xx from the API, an uncaught page error, or
 * a console.error. The journey fails at teardown if anything was collected.
 */
export function watchForUnexpectedErrors(
  context: BrowserContext,
  expected: ExpectedResponse[],
) {
  const watch = startWatching(expected);
  context.on("response", watch.onResponse);
  context.on("console", watch.onConsole);
  context.on("weberror", (webError) =>
    watch.onPageError(webError.error(), webError.page()?.url()),
  );
  return { collect: watch.collect };
}

/**
 * The same guard on one page of a context, so what it collects is that
 * page's alone, e.g. one page of a crawl that shares a context.
 */
export function watchPageForUnexpectedErrors(
  page: Page,
  expected: ExpectedResponse[],
) {
  const watch = startWatching(expected);
  page.on("response", watch.onResponse);
  page.on("console", watch.onConsole);
  page.on("pageerror", (error) => watch.onPageError(error, page.url()));
  return { collect: watch.collect };
}

function startWatching(expected: ExpectedResponse[]) {
  const pending: Promise<string | null>[] = [];
  return {
    onResponse(response: Response) {
      if (!isApiResponse(response) || response.status() < 400) return;
      if (isExpected(response, expected)) return;
      if (isKnownNoise({ kind: "response", response })) return;
      pending.push(
        isMissingActivation(response).then((missing) =>
          missing ? null : describeResponse(response),
        ),
      );
    },
    onConsole(message: ConsoleMessage) {
      if (message.type() !== "error") return;
      if (isKnownNoise({ kind: "console", text: message.text() })) return;
      pending.push(Promise.resolve(describeConsole(message)));
    },
    onPageError(error: Error, url: string | undefined) {
      if (isKnownNoise({ kind: "pageerror", text: error.message })) return;
      pending.push(
        Promise.resolve(
          `pageerror on ${url ?? "unknown page"}: ` +
            `${error.message}\n${error.stack ?? ""}`.trim(),
        ),
      );
    },
    // Until nothing is left pending (an error can arrive while an earlier one
    // is still being described), or for at most `withinMs`: a page that keeps
    // failing would otherwise keep it collecting.
    async collect(withinMs = Infinity) {
      const giveUpAt = Date.now() + withinMs;
      let settled = 0;
      const problems: (string | null)[] = [];
      while (settled < pending.length && Date.now() < giveUpAt) {
        const batch = pending.slice(settled);
        settled = pending.length;
        problems.push(...(await Promise.all(batch)));
      }
      if (pending.length > settled) {
        problems.push(
          `${pending.length - settled} more error(s) arrived after the guard stopped collecting`,
        );
      }
      return problems.filter((problem): problem is string => problem !== null);
    },
  };
}

// A body that never finishes mustn't hold the guard up: the status, method
// and URL say enough.
export const BODY_READ_TIMEOUT_MS = 5_000;

function withinBodyReadTimeout<T>(read: Promise<T>, fallback: T) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timedOut = new Promise<T>((resolve) => {
    timer = setTimeout(() => resolve(fallback), BODY_READ_TIMEOUT_MS);
  });
  return Promise.race([read, timedOut]).finally(() => clearTimeout(timer));
}

function isApiResponse(response: Response) {
  const { pathname } = new URL(response.url());
  return pathname.startsWith("/api/");
}

function isExpected(response: Response, expected: ExpectedResponse[]) {
  const method = response.request().method();
  return expected.some(
    (entry) =>
      entry.status === response.status() &&
      (!entry.method || entry.method.toUpperCase() === method) &&
      (!entry.header ||
        response.headers()[entry.header.name.toLowerCase()] ===
          entry.header.value) &&
      (typeof entry.url === "string"
        ? response.url().includes(entry.url)
        : entry.url.test(response.url())),
  );
}

type NoiseCandidate =
  | { kind: "response"; response: Response }
  | { kind: "console" | "pageerror"; text: string };

function isKnownNoise(candidate: NoiseCandidate) {
  return KNOWN_BENIGN.some((noise) => {
    if (noise.kind !== candidate.kind) return false;
    if (candidate.kind === "response") {
      return (
        (noise.status === undefined ||
          noise.status === candidate.response.status()) &&
        (!noise.url || noise.url.test(candidate.response.url()))
      );
    }
    return !noise.message || noise.message.test(candidate.text);
  });
}

// The optional recovery lookup returns this exact 404 when no activation exists.
// Different errors, methods, endpoints and response bodies remain findings.
async function isMissingActivation(response: Response) {
  if (
    response.status() !== 404 ||
    response.request().method() !== "GET" ||
    new URL(response.url()).pathname !==
      "/api/proxy/api/credits/pro-activation/current"
  )
    return false;
  const body: unknown = await withinBodyReadTimeout(
    response.json().catch(() => null),
    null,
  );
  return (
    body !== null &&
    typeof body === "object" &&
    Object.keys(body).length === 1 &&
    "detail" in body &&
    body.detail === "No activation is available"
  );
}

async function describeResponse(response: Response) {
  const body = await withinBodyReadTimeout(
    response
      .text()
      .then((text) => text.slice(0, BODY_SNIPPET_LENGTH))
      .catch(() => "<body unavailable>"),
    "<body still arriving>",
  );
  return (
    `${response.status()} ${response.request().method()} ${response.url()}\n` +
    `  body: ${body}`
  );
}

function describeConsole(message: ConsoleMessage) {
  const { url, lineNumber } = message.location();
  return `console.error at ${url}:${lineNumber}: ${message.text()}`;
}
