import type {
  BrowserContext,
  ConsoleMessage,
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
}

interface KnownNoise {
  reason: string;
  kind: "response" | "console" | "pageerror";
  status?: number;
  url?: RegExp;
  message?: RegExp;
}

// Reviewed, app-wide noise that is NOT a bug. Every entry needs a reason a
// reviewer can check. A real defect never goes here: fix it, or mark the
// journey with test.fail() and a link to the issue.
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
  const pending: Promise<string>[] = [];

  context.on("response", (response) => {
    if (!isApiResponse(response) || response.status() < 400) return;
    if (isExpected(response, expected)) return;
    if (isKnownNoise({ kind: "response", response })) return;
    pending.push(describeResponse(response));
  });

  context.on("console", (message) => {
    if (message.type() !== "error") return;
    if (isKnownNoise({ kind: "console", text: message.text() })) return;
    pending.push(Promise.resolve(describeConsole(message)));
  });

  context.on("weberror", (webError) => {
    const error = webError.error();
    if (isKnownNoise({ kind: "pageerror", text: error.message })) return;
    pending.push(
      Promise.resolve(
        `pageerror on ${webError.page()?.url() ?? "unknown page"}: ` +
          `${error.message}\n${error.stack ?? ""}`.trim(),
      ),
    );
  });

  return {
    async collect() {
      return Promise.all(pending);
    },
  };
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

async function describeResponse(response: Response) {
  const body = await response
    .text()
    .then((text) => text.slice(0, BODY_SNIPPET_LENGTH))
    .catch(() => "<body unavailable>");
  return (
    `${response.status()} ${response.request().method()} ${response.url()}\n` +
    `  body: ${body}`
  );
}

function describeConsole(message: ConsoleMessage) {
  const { url, lineNumber } = message.location();
  return `console.error at ${url}:${lineNumber}: ${message.text()}`;
}
