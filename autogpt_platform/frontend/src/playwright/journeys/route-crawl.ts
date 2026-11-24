import type {
  Browser,
  BrowserContext,
  Locator,
  Page,
  Request,
  Route,
} from "@playwright/test";
import { markChangelogSeen } from "../utils/changelog";
import { serveHostedFlags, test } from "./fixtures";
import {
  BODY_READ_TIMEOUT_MS,
  type ExpectedResponse,
  watchPageForUnexpectedErrors,
} from "./guard";
import { authHeaders, type JourneyUser } from "./journey-user";
import {
  existsInPublic,
  inAPublicFolder,
  missingPublicImage,
} from "./public-images";
import { ROUTE_RULES, type Who } from "./route-table";
import { JOURNEY_USER_AGENT } from "./user-agent";

// The route crawl's visits. Each page opens in a tab of its own, watched by
// the journey guard on that tab alone, so everything wrong is put down to the
// page that did it; the crawl reports every page's problems rather than
// stopping at the first. A session's tabs share one browser context, so a
// test opens two or three contexts rather than one per page.

type StorageState = Awaited<ReturnType<BrowserContext["storageState"]>>;

export interface Session {
  who: Who;
  /** The signed-in browser's cookies; a visitor has none. */
  storageState?: StorageState;
  /** Signs in again for a page that ends the session it is opened with. */
  user?: JourneyUser;
}

/** A dynamic route filled in with data the journey has. */
export interface Sample {
  path: string;
  shows?: string;
  expectedResponses?: ExpectedResponse[];
  /** The browser's clock runs this far ahead of the server's. */
  clockAheadMs?: number;
}

export interface Visit {
  route: string;
  path: string;
  session: Session;
  landsOn: string;
  shows?: string;
  showsErrorCard: boolean;
  expectedResponses: ExpectedResponse[];
  ownSession: boolean;
  clockAheadMs?: number;
}

export interface VisitResult {
  visit: Visit;
  problems: string[];
  ms: number;
  /** False for a page the crawl ran out of time for: it has no problems
   *  and took no time, as it was never opened. */
  reached: boolean;
}

export interface CrawlEnv {
  browser: Browser;
  baseURL: string | undefined;
  flagOverrides: Record<string, unknown>;
  /** When the test started, or a little before: its timeout runs from then. */
  startedAt: number;
}

// Pages loaded at once. The journeys run four workers on a 4-vCPU runner
// beside the whole stack, so a crawl keeps to two.
const TABS = 2;
const VISIT_TIMEOUT_MS = 30_000;
const GUARD_COLLECT_MS = 10_000;
// One batch of the guard's body reads past its deadline: the guard checks
// the deadline only between batches, and a batch's reads run together, each
// giving up after BODY_READ_TIMEOUT_MS. Twice that leaves room for
// describing what they read.
const GUARD_BATCH_OVERRUN_MS = 2 * BODY_READ_TIMEOUT_MS;
// The longest a visit runs once it has started: its checks, signing in again
// included; its guard, which can run one batch over its deadline; and
// opening and closing its tab.
const VISIT_MAX_MS =
  VISIT_TIMEOUT_MS + GUARD_COLLECT_MS + GUARD_BATCH_OVERRUN_MS + 5_000;
// What the test does after its last visit: closing the crawl's browser
// contexts and attaching its report. Fixtures tear down on a timeout of
// their own.
const REPORT_MS = 5_000;
const GOTO_TIMEOUT_MS = 30_000;
const LANDING_TIMEOUT_MS = 20_000;
const LOAD_TIMEOUT_MS = 15_000;
const IDLE_TIMEOUT_MS = 3_000;
// A page's own requests have settled once none has been open for this long.
const API_QUIET_MS = 750;
const API_SETTLE_TIMEOUT_MS = 10_000;
const STALLED_REQUEST_MS = 15_000;
const CONTENT_TIMEOUT_MS = 15_000;
const SHOWS_TIMEOUT_MS = 10_000;
const POLL_MS = 150;

export function planVisit(
  route: string,
  session: Session,
  sample?: Sample,
): Visit {
  const rule = ROUTE_RULES[route];
  const path = sample?.path ?? rule?.visit ?? route;
  const signedIn = session.who !== "visitor";
  return {
    route,
    path,
    session,
    landsOn:
      (signedIn ? rule?.signedInLandsOn : undefined) ??
      rule?.landsOn ??
      pathnameOf(path),
    shows: sample?.shows ?? rule?.shows,
    showsErrorCard: rule?.showsErrorCard ?? false,
    expectedResponses: [
      ...(rule?.expectedResponses ?? []),
      ...(sample?.expectedResponses ?? []),
    ],
    ownSession: rule?.ownSession ?? false,
    clockAheadMs: sample?.clockAheadMs,
  };
}

/**
 * Visits every page, a few at a time, and returns what each one got wrong.
 * It starts no visit too late for the visit, and the test's report, to end
 * inside the test's timeout: a page it runs out of time for comes back
 * unreached.
 */
export async function crawl(
  env: CrawlEnv,
  visits: Visit[],
): Promise<VisitResult[]> {
  const contexts = new Map<Session, Promise<BrowserContext>>();
  function sharedContext(session: Session) {
    const context = contexts.get(session) ?? openContext(env, session);
    contexts.set(session, context);
    return context;
  }
  const stopStartingAt = visitCutoff(env.startedAt, test.info().timeout);
  const results: VisitResult[] = new Array(visits.length);
  let next = 0;
  async function tab() {
    while (next < visits.length) {
      const index = next++;
      const visit = visits[index];
      results[index] =
        Date.now() < stopStartingAt
          ? await visitOnce(env, visit, sharedContext)
          : { visit, problems: [], ms: 0, reached: false };
    }
  }
  try {
    await Promise.all(Array.from({ length: TABS }, tab));
  } finally {
    await Promise.allSettled(
      [...contexts.values()].map(async (context) => (await context).close()),
    );
  }
  return results;
}

// The last moment a visit may start, for a test that started at `startedAt`
// and times out after `timeout` (0: never).
function visitCutoff(startedAt: number, timeout: number) {
  return timeout > 0
    ? startedAt + timeout - VISIT_MAX_MS - REPORT_MS
    : Infinity;
}

/** Each page that didn't load cleanly, with what went wrong on it. */
export function describeFailures(results: VisitResult[]) {
  return results
    .filter((result) => result.problems.length > 0)
    .map(
      ({ visit, problems }) =>
        `${visit.path} (${visit.session.who}):\n` +
        problems.map((problem) => `  - ${indent(problem)}`).join("\n"),
    );
}

/** Each visit's page, whom it was for, whether the crawl reached it, how
 *  long it took and what it got wrong. */
export function summarize(results: VisitResult[]) {
  return results.map(({ visit, problems, ms, reached }) => ({
    route: visit.route,
    path: visit.path,
    who: visit.session.who,
    reached,
    ms,
    problems,
  }));
}

async function visitOnce(
  env: CrawlEnv,
  visit: Visit,
  sharedContext: (session: Session) => Promise<BrowserContext>,
): Promise<VisitResult> {
  const started = Date.now();
  // A page that ends its session, or runs on a clock of its own, gets a
  // context to itself.
  const ownContext = visit.ownSession || Boolean(visit.clockAheadMs);
  const problems: string[] = [];
  let context: BrowserContext | undefined;
  let page: Page | undefined;
  let guard: ReturnType<typeof watchPageForUnexpectedErrors> | undefined;
  try {
    context = ownContext
      ? await openContext(env, visit.session, visit.clockAheadMs)
      : await sharedContext(visit.session);
    page = await context.newPage();
    guard = watchPageForUnexpectedErrors(page, visit.expectedResponses);
    problems.push(...(await checkWithinTimeout(page, visit)));
  } catch (error) {
    problems.push(`the visit broke off: ${firstLine(error)}`);
  }
  if (guard) problems.push(...(await guard.collect(GUARD_COLLECT_MS)));
  await (ownContext ? context?.close() : page?.close());
  return { visit, problems, ms: Date.now() - started, reached: true };
}

// The checks, after signing in again for a page that ends its session, are
// stopped by closing the page once they run past VISIT_TIMEOUT_MS, and
// waited for, so none outlives its visit.
async function checkWithinTimeout(page: Page, visit: Visit) {
  const checks = signInAndCheck(page, visit);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timedOut = new Promise<"timed out">((resolve) => {
    timer = setTimeout(() => resolve("timed out"), VISIT_TIMEOUT_MS);
  });
  const outcome = await Promise.race([checks, timedOut]).finally(() =>
    clearTimeout(timer),
  );
  if (outcome !== "timed out") return outcome;
  await page.close();
  await checks.catch(() => {});
  return [`its checks took over ${VISIT_TIMEOUT_MS / 1000} s`];
}

async function signInAndCheck(page: Page, visit: Visit) {
  if (visit.ownSession) await signInAgain(page, visit);
  return checkPage(page, visit);
}

async function openContext(
  env: CrawlEnv,
  session: Session,
  clockAheadMs?: number,
) {
  const context = await env.browser.newContext({
    baseURL: env.baseURL,
    bypassCSP: true,
    userAgent: JOURNEY_USER_AGENT,
    storageState: session.storageState ?? { cookies: [], origins: [] },
  });
  try {
    await markChangelogSeen(context);
    await serveHostedFlags(context, env.flagOverrides);
    await context.route(/[?&]_rsc=/, answerPrefetchesEmpty);
    await context.route(/\/_next\/image\?/, answerMissingImages);
    if (clockAheadMs) {
      await context.clock.install({ time: Date.now() + clockAheadMs });
    }
    return context;
  } catch (error) {
    await context.close();
    throw error;
  }
}

// What "loads cleanly" means, in order: the document answers, the page ends
// up where it should, its requests settle, its content shows, and none of
// the app's error UI does. The guard adds what the browser saw.
async function checkPage(page: Page, visit: Visit) {
  const problems: string[] = [];
  const failedDocuments = recordFailedDocuments(page);
  const failedImages = recordFailedImages(page);
  const requests = trackAPIRequests(page);

  await page.goto(visit.path, {
    waitUntil: "domcontentloaded",
    timeout: GOTO_TIMEOUT_MS,
  });
  // Only the URL: an image that never loads would hold back "load".
  const landed = await page
    .waitForURL((url) => url.pathname === visit.landsOn, {
      timeout: LANDING_TIMEOUT_MS,
      waitUntil: "commit",
    })
    .then(
      () => true,
      () => false,
    );
  if (!landed) {
    problems.push(
      `it should end up on ${visit.landsOn}, but it is on ${pathnameOf(page.url())}`,
    );
  }
  await settle(page, requests);

  const content = await waitForContent(page);
  if (content !== "ready")
    problems.push(`its content never showed: ${content}`);
  if (visit.shows && !(await isShown(page.getByText(visit.shows)))) {
    problems.push(`it doesn't show "${visit.shows}"`);
  }
  problems.push(...(await errorUIProblems(page, visit)));

  if (landed && pathnameOf(page.url()) !== visit.landsOn) {
    problems.push(
      `it reached ${visit.landsOn}, then moved on to ${pathnameOf(page.url())}`,
    );
  }
  problems.push(...failedDocuments, ...failedImages);
  for (const image of missingImages.get(page) ?? []) {
    problems.push(`it asks for the image ${image}, which public/ doesn't have`);
  }
  for (const request of requests.stalled()) {
    problems.push(
      `its request ${request} was still open after ${STALLED_REQUEST_MS / 1000} s`,
    );
  }
  return problems;
}

// The server's HTML shows a page's content before React has hydrated it, and
// what breaks in hydration (a mismatch, an error thrown while rendering) only
// shows after. So the checks wait for the page's scripts to have run ("load",
// which async chunks hold back), its API calls to go quiet, and the browser
// to be idle. That usually means React's render work is done, but it's a
// heuristic, not a hydration signal. No wait fails a visit: each is capped,
// and the checks after them report what's wrong.
async function settle(
  page: Page,
  requests: ReturnType<typeof trackAPIRequests>,
) {
  await page
    .waitForLoadState("load", { timeout: LOAD_TIMEOUT_MS })
    .catch(() => {});
  await requests.settle();
  await page
    .evaluate(
      (timeout) =>
        new Promise<void>((resolve) =>
          requestIdleCallback(() => resolve(), { timeout }),
        ),
      IDLE_TIMEOUT_MS,
    )
    .catch(() => {});
}

// Next prefetches the pages a page links to, rendering each on the server.
// The crawl opens every page itself, so it answers prefetches with nothing,
// which Next takes as "load that page in full if it's opened". That spares
// the stack a dozen renders per visit; admin-navigation-journey.spec.ts sends
// the admin pages' prefetches through. Aborting them instead would have Next
// log a console.error for each.
async function answerPrefetchesEmpty(route: Route) {
  if (route.request().headers()["next-router-prefetch"]) {
    await route.fulfill({ status: 204 });
  } else {
    await route.fallback();
  }
}

// Images each page asked for that public/ doesn't have.
const missingImages = new WeakMap<Page, Set<string>>();

// Next's image optimizer logs an error for an image public/ doesn't have,
// and the leg's check of the server log then fails without naming a page.
// The crawl answers those requests itself, with a 404, and reports each on
// the page that asked.
async function answerMissingImages(route: Route) {
  const image = missingPublicImage(route.request().url());
  if (!image) return route.fallback();
  const page = pageOf(route.request());
  if (page) {
    missingImages.set(page, (missingImages.get(page) ?? new Set()).add(image));
  }
  await route.fulfill({ status: 404 });
}

function pageOf(request: Request) {
  try {
    return request.frame().page();
  } catch {
    // A service worker's request has no frame.
    return undefined;
  }
}

async function signInAgain(page: Page, visit: Visit) {
  const user = visit.session.user;
  if (!user) {
    throw new Error(`${visit.route} needs a session of its own: no user given`);
  }
  // Closing the page doesn't end a request of its context, so it ends by
  // itself within the time its visit has.
  const response = await page.request.post("/api/auth/sign-in/email", {
    headers: authHeaders(),
    data: { email: user.email, password: user.password },
    timeout: VISIT_TIMEOUT_MS,
  });
  if (!response.ok()) {
    throw new Error(`signing in again failed: ${response.status()}`);
  }
}

// A page or redirect the server answered with an error, such as notFound()'s
// 404. Each is reported once the visit's checks are done.
function recordFailedDocuments(page: Page) {
  const failed: string[] = [];
  page.on("response", (response) => {
    const request = response.request();
    if (!request.isNavigationRequest() || response.status() < 400) return;
    if (!isMainFrame(page, request)) return;
    failed.push(
      `the server answered ${response.status()} for ${pathnameOf(response.url())}`,
    );
  });
  return failed;
}

// An image public/ has that the optimizer still refused, e.g. at a width it
// doesn't allow. The browser only logs it, which the guard leaves out.
function recordFailedImages(page: Page) {
  const failed: string[] = [];
  page.on("response", (response) => {
    if (response.status() < 400) return;
    const { pathname, searchParams } = new URL(response.url());
    const image = searchParams.get("url");
    if (pathname !== "/_next/image" || !image) return;
    if (!inAPublicFolder(image) || !existsInPublic(image)) return;
    failed.push(
      `the image optimizer answered ${response.status()} for ${image}`,
    );
  });
  return failed;
}

function isMainFrame(page: Page, request: Request) {
  try {
    return request.frame() === page.mainFrame();
  } catch {
    // A service worker's request has no frame.
    return false;
  }
}

// The page's own API calls, so the checks run once its data has arrived (or
// failed). Streams stay open by design and are left out.
function trackAPIRequests(page: Page) {
  // Each open request, and when it started.
  const open = new Map<Request, number>();
  let lastChange = Date.now();
  function isTracked(request: Request) {
    const { pathname } = new URL(request.url());
    return pathname.startsWith("/api/") && !/\/stream\/?$/.test(pathname);
  }
  function close(request: Request) {
    if (open.delete(request)) lastChange = Date.now();
  }
  page.on("request", (request) => {
    if (!isTracked(request)) return;
    open.set(request, Date.now());
    lastChange = Date.now();
  });
  page.on("requestfinished", close);
  page.on("requestfailed", close);
  return {
    // Not a check: a page that polls may never go quiet. It only times them.
    async settle() {
      const giveUpAt = Date.now() + API_SETTLE_TIMEOUT_MS;
      while (Date.now() < giveUpAt && !page.isClosed()) {
        if (open.size === 0 && Date.now() - lastChange >= API_QUIET_MS) return;
        await delay(POLL_MS);
      }
    },
    /** The requests open longer than STALLED_REQUEST_MS: whatever waits on
     *  them never got its data, though the page may look loaded. */
    stalled() {
      const now = Date.now();
      return [...open]
        .filter(([, since]) => now - since > STALLED_REQUEST_MS)
        .map(([request]) => `${request.method()} ${pathnameOf(request.url())}`);
    },
  };
}

async function waitForContent(page: Page) {
  const giveUpAt = Date.now() + CONTENT_TIMEOUT_MS;
  let state = await contentState(page);
  while (state !== "ready" && Date.now() < giveUpAt && !page.isClosed()) {
    await delay(POLL_MS);
    state = await contentState(page);
  }
  return state;
}

async function contentState(page: Page) {
  try {
    return await page.evaluate(readContentState);
  } catch (error) {
    // Mid-navigation the page has no document to read.
    return `the page couldn't be read (${firstLine(error)})`;
  }
}

// Runs in the page, so it uses nothing from this file. "ready" once the page
// shows text of its own, outside the sidebars and navigation, that isn't a
// loading message, and no full-page spinner covers it.
function readContentState() {
  const notContent =
    'aside, nav, [data-slot="sidebar"], [data-sidebar="sidebar"], ' +
    "[data-sonner-toaster], script, style, noscript, template";
  const visible = {
    checkOpacity: true,
    checkVisibilityCSS: true,
  };
  const cover = document.querySelector('[data-slot="loading-spinner-cover"]');
  if (cover && cover.checkVisibility(visible)) return "a full-page spinner";
  const texts: string[] = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent?.trim();
    const parent = node.parentElement;
    if (!text || !parent || parent.closest(notContent)) continue;
    if (parent.checkVisibility(visible)) texts.push(text);
  }
  const content = texts.join(" ");
  if (!content) return "no text outside the navigation";
  if (/^loading\W*$/i.test(content)) return `only "${content}"`;
  return "ready";
}

const ERROR_CARD = '[data-slot="error-card"]';

// The app's error UI: the error card (the error boundaries, global-error and
// the platform's error.tsx all show it, as do pages whose data failed), the
// root error.tsx, Next's not-found page and its last-resort exception page.
const ERROR_PAGES: { what: string; find: (page: Page) => Locator }[] = [
  {
    what: "the error page",
    find: (page) => page.getByText("Oops, something went wrong!"),
  },
  {
    what: "the not-found page",
    find: (page) => page.getByText("This page could not be found."),
  },
  {
    what: "Next.js's exception page",
    find: (page) =>
      page.getByText(/Application error: a (client|server)-side exception/),
  },
];

async function errorUIProblems(page: Page, visit: Visit) {
  const problems: string[] = [];
  const card = page.locator(ERROR_CARD);
  if (visit.showsErrorCard) {
    if (!(await isShown(card))) problems.push("it doesn't show its error card");
  } else if (await isVisibleNow(card)) {
    problems.push(
      `it shows the error card: ${await textOf(visibleOnly(card))}`,
    );
  }
  for (const { what, find } of ERROR_PAGES) {
    if (await isVisibleNow(find(page))) problems.push(`it shows ${what}`);
  }
  return problems;
}

// A copy of the text kept hidden (a phone layout's, say) doesn't count.
function visibleOnly(locator: Locator) {
  return locator.filter({ visible: true }).first();
}

function isShown(locator: Locator) {
  return visibleOnly(locator)
    .waitFor({ state: "visible", timeout: SHOWS_TIMEOUT_MS })
    .then(
      () => true,
      () => false,
    );
}

function isVisibleNow(locator: Locator) {
  return visibleOnly(locator).isVisible();
}

async function textOf(locator: Locator) {
  const text = await locator.innerText().catch(() => "");
  return JSON.stringify(text.replace(/\s+/g, " ").trim().slice(0, 300));
}

function pathnameOf(url: string) {
  return new URL(url, "http://localhost").pathname;
}

function firstLine(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return message.split("\n")[0];
}

function indent(text: string) {
  return text.replace(/\n/g, "\n    ");
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
