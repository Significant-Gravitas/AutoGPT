import type { Page } from "@playwright/test";
import {
  adminGrantUnavailable,
  grantAdminRole,
  revokeAdminRole,
} from "./admin-role";
import { type Expert, waitForSetup } from "./experts";
import { expect, test as journeyTest } from "./fixtures";
import {
  isLocalStack,
  type JourneyUser,
  provisionNewJourneyUser,
} from "./journey-user";
import { pinKnownFailure } from "./known-failure";
import {
  crawl,
  type CrawlEnv,
  describeFailures,
  planVisit,
  type Session,
  summarize,
  type Visit,
  type VisitResult,
} from "./route-crawl";
import {
  catalogName,
  hireAnExpert,
  letGo,
  makeWorkflow,
  type RouteSample,
  sampleMarketplace,
  shareAChat,
  shareARun,
  submitAListing,
  type Unsampled,
} from "./route-samples";
import {
  accessOf,
  DYNAMIC_ROUTES,
  hasKnownBug,
  INTERMITTENT_BUGS,
  isDynamic,
  KNOWN_BUGS,
  type KnownBug,
  listAppRoutes,
  ROUTE_RULES,
  type Sampler,
  samplingGaps,
  type Who,
} from "./route-table";

// Every page in the app loads cleanly for the person allowed to see it: the
// server answers it, it ends up where it should, its content shows, no error
// UI does, and the journey guard sees no failed API call, page error or
// console.error. The pages come from src/app (route-table.ts), so a new one is
// crawled as soon as it exists. Each test reports every page that failed and
// why, and notes each page it ran out of time for. The full-stack legs' check
// of the Next.js server log then covers every page too.

// Pages per test. Two load at a time, in 3 to 12 s each on a CI runner
// that four workers share with the stack.
const PAGES_PER_TEST = 8;
const ISSUES = "https://github.com/Significant-Gravitas/AutoGPT/issues";
// No setup link looks like this: the page's state for a lapsed one.
const UNKNOWN_LINK_TOKEN = "route-crawl-unknown-link";

const ROUTES = listAppRoutes();
const STATIC_ROUTES = ROUTES.filter((route) => !isDynamic(route));
const DYNAMIC = ROUTES.filter(isDynamic);

// A page with an open bug is crawled only in a test of its own per bug, as
// whom the bug shows for: its group would fail on the bug, or now and then.
function crawledInGroup(route: string) {
  return !hasKnownBug(route);
}

const PUBLIC_SLICES = slices(
  STATIC_ROUTES.filter(
    (route) => accessOf(route) === "public" && crawledInGroup(route),
  ),
);
const SIGNED_IN_SLICES = slices(
  STATIC_ROUTES.filter(
    (route) => accessOf(route) !== "admin" && crawledInGroup(route),
  ),
);
const ADMIN_SLICES = slices(
  STATIC_ROUTES.filter(
    (route) => accessOf(route) === "admin" && crawledInGroup(route),
  ),
);
const DAY_MS = 24 * 60 * 60 * 1000;
// Tries at a pinned bug's visit before its pin counts it as fixed. The crawl
// starts none too late to end inside the test's timeout.
const REPRODUCE_TRIES = 3;

// Dynamic routes nothing here samples: a page added later lands here, with
// an annotation, until it gets a sample.
const NO_SAMPLE = DYNAMIC.filter((route) => !(route in DYNAMIC_ROUTES));

interface CrawlFixtures {
  /** What the crawl needs from the test. */
  crawlEnv: CrawlEnv;
  /** Signs a new user into the test's page and makes them a platform admin,
   *  on the first call; later calls get the same admin. The role is taken
   *  back when the test ends. */
  asAdmin: () => Promise<Session>;
  /** An expert the journey user hired, its setup maybe still running, or
   *  null when the roster has none to hire here. */
  hiredExpert: Expert | null;
}

// Playwright sets up the journeys' own automatic fixtures before crawlEnv,
// on the test's time: a browser context and page with JS coverage, the
// changelog route and the guard, and for a worker's first test the browser
// itself. So the crawl takes the test as started this much earlier.
const SET_UP_BEFORE_CRAWL_ENV_MS = 5_000;

// What a test changes on the stack (an admin role, a hire) is undone in a
// fixture's teardown, which runs even after the test times out: Playwright
// abandons a finally in the test's body.
const test = journeyTest.extend<CrawlFixtures>({
  // Auto, so it is set up before the test's own fixtures, such as the
  // journey user's sign-up.
  crawlEnv: [
    async ({ browser, baseURL, flagOverrides }, provide) => {
      await provide({
        browser,
        baseURL,
        flagOverrides,
        startedAt: Date.now() - SET_UP_BEFORE_CRAWL_ENV_MS,
      });
    },
    { auto: true },
  ],
  asAdmin: async ({ page }, provide) => {
    let admin: JourneyUser | undefined;
    let signedIn: Promise<Session> | undefined;
    async function signInAnAdmin() {
      admin = await provisionNewJourneyUser(page);
      await grantAdminRole(page, admin);
      return sessionOf(page, "admin", admin);
    }
    try {
      await provide(() => (signedIn ??= signInAnAdmin()));
    } finally {
      // A test that timed out mid-grant leaves the grant running: the role
      // is taken back once it is done.
      await signedIn?.catch(() => undefined);
      if (admin) await revokeAdminRole(admin);
    }
  },
  hiredExpert: async ({ page, journeyUser: _hirer }, provide) => {
    const hired = await hireAnExpert(page);
    try {
      await provide(hired);
    } finally {
      const leftOver = hired ? await letGo(page, hired) : null;
      expect(
        leftOver,
        `cleaning up the hired expert ${hired?.id ?? ""}`.trim(),
      ).toBeNull();
    }
  },
});

test.describe("route crawl journey", () => {
  // The crawl starts no visit too late to end, with the test's report,
  // inside this timeout, and notes the pages it ran out of time for rather
  // than fail on them (route-crawl.ts).
  test.describe.configure({ timeout: 120_000 });

  for (const [index, pages] of PUBLIC_SLICES.entries()) {
    test(`public pages load for a signed-out visitor (${index + 1}/${PUBLIC_SLICES.length}): ${pages[0]} to ${pages.at(-1)}`, async ({
      crawlEnv,
      page,
    }) => {
      await expectAllClean(
        await crawlPages(crawlEnv, page, pages, { who: "visitor" }),
      );
    });
  }

  for (const [index, pages] of SIGNED_IN_SLICES.entries()) {
    test(`signed-in pages load (${index + 1}/${SIGNED_IN_SLICES.length}): ${pages[0]} to ${pages.at(-1)}`, async ({
      crawlEnv,
      page,
      journeyUser,
    }) => {
      const user = await sessionOf(page, "user", journeyUser);
      await expectAllClean(await crawlPages(crawlEnv, page, pages, user));
    });
  }

  test.describe("as an admin", () => {
    const noAdmin = adminGrantUnavailable();
    test.skip(Boolean(noAdmin), noAdmin);

    for (const [index, pages] of ADMIN_SLICES.entries()) {
      // The first slice also previews a listing the admin submits.
      const withPreview = index === 0;
      test(`admin pages load for an admin (${index + 1}/${ADMIN_SLICES.length}): ${pages[0]} to ${pages.at(-1)}${withPreview ? ", and a listing's preview" : ""}`, async ({
        crawlEnv,
        page,
        asAdmin,
      }) => {
        const admin = await asAdmin();
        const preview = withPreview ? await previewAListing(page, admin) : [];
        await expectAllClean(
          await crawlPages(crawlEnv, page, pages, admin, preview),
        );
      });
    }
  });

  test("the Marketplace's listing, creator, expert and skill pages load, signed out and signed in", async ({
    crawlEnv,
    page,
    journeyUser,
  }) => {
    const { agent, creator, expert, skill } = await sampleMarketplace(page);
    const user = await sessionOf(page, "user", journeyUser);
    const visits = planSamples(
      "marketplace",
      [
        agent ??
          unsampled(
            "/marketplace/agent/[creator]/[slug]",
            "no approved listing",
          ),
        creator ??
          unsampled("/marketplace/creator/[creator]", "no approved listing"),
        expert ??
          unsampled("/marketplace/experts/[expertId]", "no roster published"),
        skill ?? unsampled("/marketplace/skills/[slug]", "no skill"),
      ],
      [{ who: "visitor" }, user],
    );
    await expectAllClean(await crawl(crawlEnv, visits));
  });

  test(
    "the user's own workflow, hired expert and a lapsed setup link load",
    {
      annotation: NO_SAMPLE.map((route) => ({
        type: "unsampled route",
        description: `${route}: no sample in route-table.ts's DYNAMIC_ROUTES yet`,
      })),
    },
    async ({ crawlEnv, page, journeyUser, hiredExpert }) => {
      const workflow = await makeWorkflow(page);
      const expert = hiredExpert && (await waitForSetup(page, hiredExpert.id));
      const user = await sessionOf(page, "user", journeyUser);
      const visits = planSamples(
        "own data",
        [
          {
            route: "/library/agents/[id]",
            path: `/library/agents/${workflow.libraryAgentID}`,
            shows: workflow.name,
          },
          expert
            ? {
                route: "/team/[expertId]",
                path: `/team/${expert.id}`,
                shows: expert.name,
              }
            : unsampled(
                "/team/[expertId]",
                "no roster expert without a schedule",
              ),
          {
            route: "/link/[token]",
            path: `/link/${UNKNOWN_LINK_TOKEN}`,
            shows: "Setup failed",
            expectedResponses: [
              {
                status: 404,
                method: "GET",
                url: `/api/platform-linking/tokens/${UNKNOWN_LINK_TOKEN}/info`,
              },
            ],
          },
        ],
        [user],
      );
      await expectAllClean(await crawl(crawlEnv, visits));
    },
  );

  test("a shared run and a shared chat load for a signed-out visitor", async ({
    crawlEnv,
    page,
    // The owner, who makes and shares them.
    journeyUser: _owner,
  }) => {
    // A chat turn starts a Claude Agent SDK session, which can take a while
    // on CI (MODEL_TURN_TIMEOUT).
    test.setTimeout(120_000);
    const [run, chat] = await Promise.all([shareARun(page), shareAChat(page)]);
    const visits = planSamples(
      "shared",
      [
        {
          route: "/share/[token]",
          path: `/share/${run.token}`,
          shows: run.graphName,
        },
        chat
          ? {
              route: "/share/chat/[token]",
              path: `/share/chat/${chat.token}`,
              shows: chat.answer,
            }
          : {
              route: "/share/chat/[token]",
              why: "its answer comes from the scripted model, which this stack doesn't have",
            },
      ],
      [{ who: "visitor" }],
    );
    await expectAllClean(await crawl(crawlEnv, visits));
  });

  // How long ago something happened reads the same in the server's HTML and
  // in the browser hydrating it, whatever the visitor's clock says (#17309,
  // #17310): these pages show an age and open with the browser's clock ahead
  // of the server's, which changes what any age reads.
  test("the admin Marketplace page loads with the admin's clock ahead of the server's", async ({
    crawlEnv,
    page,
    asAdmin,
  }) => {
    const noAdmin = adminGrantUnavailable();
    test.skip(Boolean(noAdmin), noAdmin || undefined);
    const admin = await asAdmin();
    // Submitted moments ago, so its age changes within the five minutes.
    const listing = await submitAListing(page);
    await expectAllClean(
      await crawl(crawlEnv, [
        planVisit("/admin/marketplace", admin, {
          path: `/admin/marketplace?search=${encodeURIComponent(listing.title)}`,
          shows: listing.title,
          clockAheadMs: 5 * 60_000,
        }),
      ]),
    );
  });

  for (const which of ["agent", "skill"] as const) {
    test(`a Marketplace ${which} page loads with the visitor's clock days ahead of the server's`, async ({
      crawlEnv,
      page,
    }) => {
      const sample = (await sampleMarketplace(page))[which];
      await expectAllClean(await visitAsVisitor(crawlEnv, sample, 2 * DAY_MS));
    });
  }

  // A skill names the MCP service it needs by the catalog entry's name. Its
  // page shows that service's logo, which public/ has (#17313), and calls the
  // service what the catalog does, not "Mcp Amplitude" (#17336).
  test("a skill whose first service is an MCP server loads with that service's logo and name", async ({
    crawlEnv,
    page,
  }) => {
    const { skillWithMCPService: skill } = await sampleMarketplace(page);
    await expectAllClean(await visitAsVisitor(crawlEnv, skill));
    // visitAsVisitor has skipped the test if the stack has no such skill.
    if (!skill) return;
    const name = await catalogName(page, skill.service);
    await page.goto(skill.path, { waitUntil: "domcontentloaded" });
    const worksWith = page
      .getByRole("main")
      .locator("header")
      .getByText(/^Works with/);
    await expect(worksWith).toContainText(name);
    await expect(worksWith).not.toContainText("Mcp ");
  });

  for (const bug of KNOWN_BUGS) {
    const { route, who, issue } = bug;
    test(
      `${route} loads for ${who === "visitor" ? "a signed-out visitor" : `the ${who}`} (known bug #${issue})`,
      { annotation: { type: "issue", description: `${ISSUES}/${issue}` } },
      async ({ crawlEnv, page, asAdmin }) => {
        const noAdmin = who === "admin" && adminGrantUnavailable();
        test.skip(Boolean(noAdmin), noAdmin || undefined);
        const reproduce = REPRODUCTIONS[reproductionKey(bug)];
        if (!reproduce && isDynamic(route)) {
          throw new Error(
            `${route} has a dynamic segment: give its pin a visit in REPRODUCTIONS`,
          );
        }
        const fixtures: PinFixtures = { crawlEnv, page, asAdmin };
        // A visit can miss a timing bug on a busy runner, so it is tried
        // again until the bug shows: the pin then fails every run while the
        // bug is there, and passes every run once it is fixed.
        const results: VisitResult[] = [];
        for (let tries = 0; tries < REPRODUCE_TRIES; tries++) {
          const [result] = reproduce
            ? await reproduce(fixtures)
            : await crawlAs(fixtures, who, [route]);
          // The crawl ran out of time for this try.
          if (!result.reached) break;
          results.push(result);
          if (result.problems.some((problem) => bug.problem.test(problem))) {
            break;
          }
        }
        await attachPages(results);
        if (results.length === 0) {
          // Noted, as a group notes a page it didn't reach.
          annotate(
            "unreached page",
            `${route} (${who}): the crawl ran out of time before it`,
          );
          return;
        }
        const problems = results.flatMap((result) => result.problems);
        // The page's other known bugs have pins of their own, or none can.
        const others = [
          ...KNOWN_BUGS.filter(
            (other) => other.route === route && other !== bug,
          ),
          ...INTERMITTENT_BUGS,
        ];
        const unrelated = problems.filter(
          (problem) =>
            !bug.problem.test(problem) &&
            !others.some((other) => other.problem.test(problem)),
        );
        expect(
          unrelated,
          `${route} should fail only on its known bugs`,
        ).toEqual([]);
        const pinned = pinKnownFailure(
          `#${issue}: ${bug.what}`,
          `${route} should load without #${issue} for the ${who}`,
        );
        expect(
          problems.filter((problem) => bug.problem.test(problem)),
          pinned,
        ).toEqual([]);
      },
    );
  }
});

/** What a pinned bug's visit is made with: the test's own fixtures. */
interface PinFixtures {
  crawlEnv: CrawlEnv;
  page: Page;
  asAdmin: () => Promise<Session>;
}

function reproductionKey({ route, issue }: KnownBug) {
  return `${route} #${issue}`;
}

// A pinned bug that a plain visit doesn't show every time gets the visit
// that does. Any other is crawled as whom the bug shows for.
const REPRODUCTIONS: Record<
  string,
  (fixtures: PinFixtures) => Promise<VisitResult[]>
> = {};

/** The Marketplace's `sample`, opened signed out, the browser's clock
 *  `clockAheadMs` ahead of the server's. */
async function visitAsVisitor(
  env: CrawlEnv,
  sample: RouteSample | undefined,
  clockAheadMs?: number,
) {
  test.skip(!sample, "this stack has nothing to open for it");
  if (!sample) return [];
  return crawl(env, [
    planVisit(sample.route, { who: "visitor" }, { ...sample, clockAheadMs }),
  ]);
}

/** Crawls the pages as `who`: signed out, a new user, or the test's admin. */
async function crawlAs(fixtures: PinFixtures, who: Who, routes: string[]) {
  const session = await sessionAs(fixtures, who);
  return crawl(
    fixtures.crawlEnv,
    routes.map((route) => planVisit(route, session)),
  );
}

async function sessionAs(
  { page, asAdmin }: PinFixtures,
  who: Who,
): Promise<Session> {
  if (who === "visitor") return { who };
  if (who === "admin") return asAdmin();
  return sessionOf(page, who, await provisionNewJourneyUser(page));
}

async function sessionOf(
  page: Page,
  who: Who,
  user: JourneyUser,
): Promise<Session> {
  return { who, user, storageState: await page.context().storageState() };
}

/** Crawls the pages this stack serves, as `session`, with `more` visits. */
async function crawlPages(
  env: CrawlEnv,
  page: Page,
  routes: string[],
  session: Session,
  more: Visit[] = [],
) {
  const served = await servedHere(page, routes);
  return crawl(env, [
    ...served.map((route) => planVisit(route, session)),
    ...more,
  ]);
}

// A page only some stacks serve is crawled where it is served, and noted
// elsewhere.
async function servedHere(page: Page, routes: string[]) {
  const paymentsOn =
    routes.some((route) => ROUTE_RULES[route]?.hostedOnly) &&
    (await takesPayments(page));
  return routes.filter((route) => {
    const why = whyNotServed(route, paymentsOn);
    if (why) annotate("page not served here", `${route}: ${why}`);
    return !why;
  });
}

function whyNotServed(route: string, paymentsOn: boolean) {
  const rule = ROUTE_RULES[route];
  if (rule?.localOnly && !isLocalStack()) return "only a local stack serves it";
  if (rule?.hostedOnly && !paymentsOn) {
    return "only a stack that takes payments serves it, and this one doesn't";
  }
  return null;
}

/**
 * Whether the stack takes payments (the backend's ENABLE_PLATFORM_PAYMENT),
 * which decides whether a self-hosted build has Billing (useBillingStatus).
 * A hosted build has Billing either way, so on one that takes no payments the
 * crawl leaves out a page that is there.
 */
async function takesPayments(page: Page) {
  const response = await page.request.get("/api/proxy/api/platform-config");
  expect(
    response.ok(),
    `reading the stack's platform config: ${response.status()}`,
  ).toBe(true);
  const config = (await response.json()) as { payment_enabled: boolean };
  return config.payment_enabled;
}

/** The admin's preview of a listing they submit. */
async function previewAListing(page: Page, admin: Session) {
  const listing = await submitAListing(page);
  return planSamples(
    "admin",
    [
      {
        route: "/admin/marketplace/preview/[id]",
        path: `/admin/marketplace/preview/${listing.versionID}`,
        shows: listing.title,
      },
    ],
    [admin],
  );
}

/**
 * The visits to the dynamic routes `sampler` samples, as each session. It
 * fails unless the test has a sample, or says why it has none, for every
 * route DYNAMIC_ROUTES gives it, so none there goes uncrawled unnoticed. A
 * route with an open bug is crawled in its pinned test instead.
 */
function planSamples(
  sampler: Sampler,
  samples: (RouteSample | Unsampled)[],
  sessions: Session[],
) {
  const { missing, misplaced } = samplingGaps(
    sampler,
    samples.map((sample) => sample.route),
  );
  expect(
    missing,
    `DYNAMIC_ROUTES has the "${sampler}" test sample these routes, but it has no sample for them, nor says why`,
  ).toEqual([]);
  expect(
    misplaced,
    `the "${sampler}" test samples these routes, which DYNAMIC_ROUTES doesn't give it`,
  ).toEqual([]);
  return samples.flatMap((sample) => {
    if ("why" in sample) {
      annotate("unsampled route", `${sample.route}: ${sample.why}`);
      return [];
    }
    if (!crawledInGroup(sample.route)) return [];
    return sessions.map((session) => planVisit(sample.route, session, sample));
  });
}

/** A dynamic route this stack has nothing to fill in with, and why. */
function unsampled(route: string, why: string): Unsampled {
  return { route, why: `${why} on this stack` };
}

function annotate(type: string, description: string) {
  test.info().annotations.push({ type, description });
}

async function attachPages(results: VisitResult[]) {
  await test.info().attach("pages.json", {
    body: JSON.stringify(summarize(results), null, 2),
    contentType: "application/json",
  });
}

async function expectAllClean(results: VisitResult[]) {
  await attachPages(results);
  // A page the crawl ran out of time for is noted, not failed: a slow
  // runner is no broken page.
  for (const { visit } of results.filter((result) => !result.reached)) {
    annotate(
      "unreached page",
      `${visit.path} (${visit.session.who}): the crawl ran out of time before it`,
    );
  }
  const reached = results.filter((result) => result.reached);
  // Notes on unreached pages are fine; a test that reached none checked
  // nothing, so a slow set-up mustn't turn it into a pass.
  expect(
    results.length === 0 || reached.length > 0,
    `the crawl ran out of time before any of its ${results.length} pages`,
  ).toBe(true);
  const failures = describeFailures(reached.map(noteKnownBugs));
  expect(
    failures.length,
    `${failures.length} of ${reached.length} pages didn't load cleanly:\n\n` +
      failures.join("\n\n"),
  ).toBe(0);
}

// Known bugs a group notes on its pages rather than fail on: one that other
// pages show too, depending on what they list, which its own test pins, and
// one that shows only now and then, which no test can pin.
function noteKnownBugs(result: VisitResult): VisitResult {
  const noted = [
    ...KNOWN_BUGS.filter((bug) => bug.showsElsewhere),
    ...INTERMITTENT_BUGS,
  ];
  const problems = result.problems.filter((problem) => {
    const bug = noted.find((known) => known.problem.test(problem));
    if (!bug) return true;
    annotate(
      "known bug seen",
      `#${bug.issue} on ${result.visit.path}: ${problem}`,
    );
    return false;
  });
  return { ...result, problems };
}

// The pages in near-equal slices of at most PAGES_PER_TEST, in route order.
function slices(routes: string[]) {
  const count = Math.max(1, Math.ceil(routes.length / PAGES_PER_TEST));
  const size = Math.ceil(routes.length / count);
  return Array.from({ length: count }, (_, index) =>
    routes.slice(index * size, (index + 1) * size),
  ).filter((slice) => slice.length > 0);
}
