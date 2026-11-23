import type { Page } from "@playwright/test";
import {
  adminGrantUnavailable,
  grantAdminRole,
  revokeAdminRole,
} from "./admin-role";
import { expect, test } from "./fixtures";
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
  type VisitResult,
} from "./route-crawl";
import { waitForSetup } from "./experts";
import {
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
  type Who,
} from "./route-table";

// Every page in the app loads cleanly for the person allowed to see it: the
// server answers it, it ends up where it should, its content shows, no error
// UI does, and the journey guard sees no failed API call, page error or
// console.error. The pages come from src/app (route-table.ts), so a new one is
// crawled as soon as it exists. Each test reports every page that failed and
// why. The full-stack legs' check of the Next.js server log then covers
// every page too.

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
// Tries at a pinned bug's visit before its pin counts it as fixed, started
// only within RETRY_WINDOW_MS so the last one ends inside the test's timeout.
const REPRODUCE_TRIES = 3;
const RETRY_WINDOW_MS = 40_000;

// Dynamic routes nothing here samples: a page added later lands here, with
// an annotation, until it gets a sample.
const NO_SAMPLE = DYNAMIC.filter((route) => !(route in DYNAMIC_ROUTES));

test.describe("route crawl journey", () => {
  // A crawl takes 10 to 55 s. Its budgets (no visit starts after 45 s, a
  // visit's checks get 30 s and its guard 10 s) bound the worst case near
  // 100 s, past the journeys' 90 s, which would lose the report.
  test.describe.configure({ timeout: 120_000 });

  for (const [index, pages] of PUBLIC_SLICES.entries()) {
    test(`public pages load for a signed-out visitor (${index + 1}/${PUBLIC_SLICES.length}): ${pages[0]} to ${pages.at(-1)}`, async ({
      browser,
      baseURL,
      flagOverrides,
    }) => {
      const env = { browser, baseURL, flagOverrides };
      const visitor: Session = { who: "visitor" };
      await expectAllClean(
        await crawl(
          env,
          pages.map((route) => planVisit(route, visitor)),
        ),
      );
    });
  }

  for (const [index, pages] of SIGNED_IN_SLICES.entries()) {
    test(`signed-in pages load (${index + 1}/${SIGNED_IN_SLICES.length}): ${pages[0]} to ${pages.at(-1)}`, async ({
      browser,
      baseURL,
      flagOverrides,
      page,
      journeyUser,
    }) => {
      const env = { browser, baseURL, flagOverrides };
      const user = await sessionOf(page, "user", journeyUser);
      await expectAllClean(
        await crawl(
          env,
          pages.map((route) => planVisit(route, user)),
        ),
      );
    });
  }

  test.describe("as an admin", () => {
    const noAdmin = adminGrantUnavailable();
    test.skip(Boolean(noAdmin), noAdmin);

    for (const [index, pages] of ADMIN_SLICES.entries()) {
      // The first slice also previews a listing the admin submits.
      const withPreview = index === 0;
      test(`admin pages load for an admin (${index + 1}/${ADMIN_SLICES.length}): ${pages[0]} to ${pages.at(-1)}${withPreview ? ", and a listing's preview" : ""}`, async ({
        browser,
        baseURL,
        flagOverrides,
        page,
      }) => {
        const env = { browser, baseURL, flagOverrides };
        const results = await asAnAdmin(page, async (admin) => {
          const visits = pages
            .filter(servedHere)
            .map((route) => planVisit(route, admin));
          if (withPreview) {
            const listing = await submitAListing(page);
            visits.push(
              planVisit("/admin/marketplace/preview/[id]", admin, {
                path: `/admin/marketplace/preview/${listing.versionID}`,
                shows: listing.title,
              }),
            );
          }
          return crawl(env, visits);
        });
        await expectAllClean(results);
      });
    }
  });

  test("the Marketplace's listing, creator, expert and skill pages load, signed out and signed in", async ({
    browser,
    baseURL,
    flagOverrides,
    page,
    journeyUser,
  }) => {
    const env = { browser, baseURL, flagOverrides };
    const { agent, creator, expert, skill } = await sampleMarketplace(page);
    const samples = [
      agent ??
        unsampled("/marketplace/agent/[creator]/[slug]", "no approved listing"),
      creator ??
        unsampled("/marketplace/creator/[creator]", "no approved listing"),
      expert ??
        unsampled("/marketplace/experts/[expertId]", "no roster published"),
      skill ?? unsampled("/marketplace/skills/[slug]", "no skill"),
    ]
      .filter((sample): sample is RouteSample => sample !== null)
      // A page with an open bug is crawled in its pinned test instead, and
      // here again once the bug is fixed and its pin dropped.
      .filter((sample) => crawledInGroup(sample.route));
    const user = await sessionOf(page, "user", journeyUser);
    const visitor: Session = { who: "visitor" };
    await expectAllClean(
      await crawl(
        env,
        samples.flatMap((sample) => [
          planVisit(sample.route, visitor, sample),
          planVisit(sample.route, user, sample),
        ]),
      ),
    );
  });

  test(
    "the user's own workflow, hired expert and a lapsed setup link load",
    {
      annotation: NO_SAMPLE.map((route) => ({
        type: "unsampled route",
        description: `${route}: no sample in route-table.ts's DYNAMIC_ROUTES yet`,
      })),
    },
    async ({ browser, baseURL, flagOverrides, page, journeyUser }) => {
      const env = { browser, baseURL, flagOverrides };
      const workflow = await makeWorkflow(page);
      const hired = await hireAnExpert(page);
      if (!hired) {
        annotateUnsampled([
          {
            route: "/team/[expertId]",
            why: "no roster expert without a schedule on this stack",
          },
        ]);
      }
      const user = await sessionOf(page, "user", journeyUser);
      let results: VisitResult[];
      try {
        const expert = hired && (await waitForSetup(page, hired.id));
        results = await crawl(env, [
          planVisit("/library/agents/[id]", user, {
            path: `/library/agents/${workflow.libraryAgentID}`,
            shows: workflow.name,
          }),
          ...(expert
            ? [
                planVisit("/team/[expertId]", user, {
                  path: `/team/${expert.id}`,
                  shows: expert.name,
                }),
              ]
            : []),
          planVisit("/link/[token]", user, {
            path: `/link/${UNKNOWN_LINK_TOKEN}`,
            shows: "Setup failed",
            expectedResponses: [
              {
                status: 404,
                method: "GET",
                url: `/api/platform-linking/tokens/${UNKNOWN_LINK_TOKEN}/info`,
              },
            ],
          }),
        ]);
      } finally {
        const leftOver = hired ? await letGo(page, hired) : null;
        expect.soft(leftOver, "cleaning up the hired expert").toBeNull();
      }
      await expectAllClean(results);
    },
  );

  test("a shared run and a shared chat load for a signed-out visitor", async ({
    browser,
    baseURL,
    flagOverrides,
    page,
    // The owner, who makes and shares them.
    journeyUser: _owner,
  }) => {
    // A chat turn starts a Claude Agent SDK session, which can take a while
    // on CI (MODEL_TURN_TIMEOUT).
    test.setTimeout(120_000);
    const env = { browser, baseURL, flagOverrides };
    const [run, chat] = await Promise.all([shareARun(page), shareAChat(page)]);
    if (!chat) {
      annotateUnsampled([
        {
          route: "/share/chat/[token]",
          why: "its answer comes from the scripted model, which this stack doesn't have",
        },
      ]);
    }
    const visitor: Session = { who: "visitor" };
    await expectAllClean(
      await crawl(env, [
        planVisit("/share/[token]", visitor, {
          path: `/share/${run.token}`,
          shows: run.graphName,
        }),
        ...(chat
          ? [
              planVisit("/share/chat/[token]", visitor, {
                path: `/share/chat/${chat.token}`,
                shows: chat.answer,
              }),
            ]
          : []),
      ]),
    );
  });

  // A skill names the MCP service it needs by the catalog entry's name, and
  // its page shows that service's logo, which public/ has (#17313).
  test("a skill whose first service is an MCP server loads with that service's logo", async ({
    browser,
    baseURL,
    flagOverrides,
    page,
  }) => {
    const env = { browser, baseURL, flagOverrides };
    const { skillWithMCPService } = await sampleMarketplace(page);
    await expectAllClean(await visitAsVisitor(env, skillWithMCPService));
  });

  for (const bug of KNOWN_BUGS) {
    const { route, who, issue } = bug;
    test(
      `${route} loads for ${who === "visitor" ? "a signed-out visitor" : `the ${who}`} (known bug #${issue})`,
      { annotation: { type: "issue", description: `${ISSUES}/${issue}` } },
      async ({ browser, baseURL, flagOverrides, page }) => {
        const noAdmin = who === "admin" && adminGrantUnavailable();
        test.skip(Boolean(noAdmin), noAdmin || undefined);
        const env = { browser, baseURL, flagOverrides };
        const reproduce = REPRODUCTIONS[reproductionKey(bug)];
        if (!reproduce && isDynamic(route)) {
          throw new Error(
            `${route} has a dynamic segment: give its pin a visit in REPRODUCTIONS`,
          );
        }
        // A visit can miss a timing bug on a busy runner, so it is tried
        // again until the bug shows: the pin then fails every run while the
        // bug is there, and passes every run once it is fixed.
        const startedAt = Date.now();
        const results: VisitResult[] = [];
        for (let tries = 0; tries < REPRODUCE_TRIES; tries++) {
          if (tries > 0 && Date.now() - startedAt > RETRY_WINDOW_MS) break;
          const [result] = reproduce
            ? await reproduce(env, page)
            : await crawlAs(env, page, who, [route]);
          results.push(result);
          if (result.problems.some((problem) => bug.problem.test(problem))) {
            break;
          }
        }
        await attachPages(results);
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

function reproductionKey({ route, issue }: KnownBug) {
  return `${route} #${issue}`;
}

// A pinned bug that a plain visit doesn't show every time gets the visit
// that does. Any other is crawled as whom the bug shows for.
const REPRODUCTIONS: Record<
  string,
  (env: CrawlEnv, page: Page) => Promise<VisitResult[]>
> = {
  // A submission's age is worked out on the server and again in the
  // browser. With the browser's clock ahead, one submitted moments ago reads
  // differently in each, every time.
  "/admin/marketplace #17309": (env, page) =>
    asAnAdmin(page, async (admin) => {
      const listing = await submitAListing(page);
      return crawl(env, [
        planVisit("/admin/marketplace", admin, {
          path: `/admin/marketplace?search=${encodeURIComponent(listing.title)}`,
          shows: listing.title,
          clockAheadMs: 5 * 60_000,
        }),
      ]);
    }),
  // The same for "Last updated" and "Updated". Two days ahead changes what
  // any age reads.
  "/marketplace/agent/[creator]/[slug] #17310": async (env, page) =>
    visitAsVisitor(env, (await sampleMarketplace(page)).agent, 2 * DAY_MS),
  "/marketplace/skills/[slug] #17310": async (env, page) =>
    visitAsVisitor(env, (await sampleMarketplace(page)).skill, 2 * DAY_MS),
};

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

/** Crawls the pages as `who`, in a session the test makes and ends. */
async function crawlAs(env: CrawlEnv, page: Page, who: Who, routes: string[]) {
  if (who === "visitor") {
    return crawl(
      env,
      routes.map((route) => planVisit(route, { who })),
    );
  }
  if (who === "admin") {
    return asAnAdmin(page, (admin) =>
      crawl(
        env,
        routes.map((route) => planVisit(route, admin)),
      ),
    );
  }
  const user = await sessionOf(page, who, await provisionNewJourneyUser(page));
  return crawl(
    env,
    routes.map((route) => planVisit(route, user)),
  );
}

/** A new user made an admin for `body`, and no longer one after it. */
async function asAnAdmin<T>(
  page: Page,
  body: (admin: Session) => Promise<T>,
): Promise<T> {
  const user = await provisionNewJourneyUser(page);
  try {
    await grantAdminRole(page, user);
    return await body(await sessionOf(page, "admin", user));
  } finally {
    await revokeAdminRole(user);
  }
}

async function sessionOf(
  page: Page,
  who: Who,
  user: JourneyUser,
): Promise<Session> {
  return { who, user, storageState: await page.context().storageState() };
}

// Pages only a local stack serves are left out elsewhere, and said so.
function servedHere(route: string) {
  if (!ROUTE_RULES[route]?.localOnly || isLocalStack()) return true;
  annotateUnsampled([{ route, why: ROUTE_RULES[route].reason }]);
  return false;
}

/** Says a sample is missing, for a list of samples to leave it out of. */
function unsampled(route: string, why: string) {
  annotateUnsampled([{ route, why: `${why} on this stack` }]);
  return null;
}

function annotateUnsampled(unsampled: Unsampled[]) {
  for (const { route, why } of unsampled) {
    test.info().annotations.push({
      type: "unsampled route",
      description: `${route}: ${why}`,
    });
  }
}

async function attachPages(results: VisitResult[]) {
  await test.info().attach("pages.json", {
    body: JSON.stringify(summarize(results), null, 2),
    contentType: "application/json",
  });
}

async function expectAllClean(results: VisitResult[]) {
  await attachPages(results);
  const failures = describeFailures(results.map(noteKnownBugs));
  expect(
    failures.length,
    `${failures.length} of ${results.length} pages didn't load cleanly:\n\n` +
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
    test.info().annotations.push({
      type: "known bug seen",
      description: `#${bug.issue} on ${result.visit.path}: ${problem}`,
    });
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
