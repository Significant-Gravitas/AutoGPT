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
import {
  hireAnExpert,
  letGo,
  makeWorkflow,
  sampleMarketplace,
  shareAChat,
  shareARun,
  submitAListing,
  type Unsampled,
} from "./route-samples";
import {
  accessOf,
  DYNAMIC_ROUTES,
  isDynamic,
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

// Pages per test. Two load at a time in about 2-4 s each on CI.
const PAGES_PER_TEST = 12;
const ISSUES = "https://github.com/Significant-Gravitas/AutoGPT/issues";
// No setup link looks like this: the page's state for a lapsed one.
const UNKNOWN_LINK_TOKEN = "route-crawl-unknown-link";

const ROUTES = listAppRoutes();
const STATIC_ROUTES = ROUTES.filter((route) => !isDynamic(route));
const DYNAMIC = ROUTES.filter(isDynamic);

// A page pinned to an open bug is crawled only in its own test, as whom the
// bug shows for: its group would fail on the bug, or now and then on it.
function crawledInGroup(route: string) {
  return !ROUTE_RULES[route]?.bug;
}

const PUBLIC_PAGES = STATIC_ROUTES.filter(
  (route) => accessOf(route) === "public" && crawledInGroup(route),
);
const SIGNED_IN_SLICES = slices(
  STATIC_ROUTES.filter(
    (route) => accessOf(route) !== "admin" && crawledInGroup(route),
  ),
);
const ADMIN_PAGES = STATIC_ROUTES.filter(
  (route) => accessOf(route) === "admin" && crawledInGroup(route),
);
const PINNED = ROUTES.flatMap((route) => {
  const bug = ROUTE_RULES[route]?.bug;
  return bug ? [{ route, bug }] : [];
});
const DAY_MS = 24 * 60 * 60 * 1000;

// Dynamic routes nothing here samples: a page added later lands here, with
// an annotation, until it gets a sample.
const NO_SAMPLE = DYNAMIC.filter((route) => !(route in DYNAMIC_ROUTES));

test.describe("route crawl journey", () => {
  test("public pages load for a signed-out visitor", async ({
    browser,
    baseURL,
    flagOverrides,
  }) => {
    const env = { browser, baseURL, flagOverrides };
    const visitor: Session = { who: "visitor" };
    await expectAllClean(
      await crawl(
        env,
        PUBLIC_PAGES.map((route) => planVisit(route, visitor)),
      ),
    );
  });

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

    test("admin pages load for an admin, with a listing's preview", async ({
      browser,
      baseURL,
      flagOverrides,
      page,
    }) => {
      const env = { browser, baseURL, flagOverrides };
      const results = await asAnAdmin(page, async (admin) => {
        const listing = await submitAListing(page);
        return crawl(env, [
          ...ADMIN_PAGES.filter(servedHere).map((route) =>
            planVisit(route, admin),
          ),
          planVisit("/admin/marketplace/preview/[id]", admin, {
            path: `/admin/marketplace/preview/${listing.versionID}`,
            shows: listing.title,
          }),
        ]);
      });
      await expectAllClean(results);
    });
  });

  test("the Marketplace's pages for one listing's creator and an expert load, signed out and signed in", async ({
    browser,
    baseURL,
    flagOverrides,
    page,
    journeyUser,
  }) => {
    const env = { browser, baseURL, flagOverrides };
    const { samples, unsampled } = await sampleMarketplace(page);
    annotateUnsampled(unsampled);
    const user = await sessionOf(page, "user", journeyUser);
    const visitor: Session = { who: "visitor" };
    await expectAllClean(
      await crawl(
        env,
        samples
          .filter((sample) => crawledInGroup(sample.route))
          .flatMap((sample) => [
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
      const [workflow, expert] = await Promise.all([
        makeWorkflow(page),
        hireAnExpert(page),
      ]);
      if (!expert) {
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
        const leftOver = expert ? await letGo(page, expert) : null;
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

  for (const { route, bug } of PINNED) {
    test(
      `${route} loads for ${bug.who === "visitor" ? "a signed-out visitor" : `the ${bug.who}`} (known bug #${bug.issue})`,
      { annotation: { type: "issue", description: `${ISSUES}/${bug.issue}` } },
      async ({ browser, baseURL, flagOverrides, page }) => {
        const noAdmin = bug.who === "admin" && adminGrantUnavailable();
        test.skip(Boolean(noAdmin), noAdmin || undefined);
        const env = { browser, baseURL, flagOverrides };
        const reproduce = REPRODUCTIONS[route];
        if (!reproduce && isDynamic(route)) {
          throw new Error(
            `${route} has a dynamic segment: give its pin a visit in REPRODUCTIONS`,
          );
        }
        const [result] = reproduce
          ? await reproduce(env, page)
          : await crawlAs(env, page, bug.who, [route]);
        const unrelated = result.problems.filter(
          (problem) => !bug.problem.test(problem),
        );
        expect(unrelated, `${route} should fail only on #${bug.issue}`).toEqual(
          [],
        );
        const pinned = pinKnownFailure(
          `#${bug.issue}: ${bug.what}`,
          `${route} should load cleanly for the ${bug.who}`,
        );
        expect(result.problems, pinned).toEqual([]);
      },
    );
  }
});

// A pinned bug that a plain visit doesn't show every time gets the visit
// that does. Any other is crawled as whom the bug shows for.
const REPRODUCTIONS: Record<
  string,
  (env: CrawlEnv, page: Page) => Promise<VisitResult[]>
> = {
  // A submission's age is worked out on the server and again in the
  // browser. With the browser's clock ahead, one submitted moments ago reads
  // differently in each, every time (#17309).
  "/admin/marketplace": (env, page) =>
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
  // The same for "Last updated" and "Updated" (#17310). Two days ahead
  // changes what any age reads.
  "/marketplace/agent/[creator]/[slug]": (env, page) =>
    visitAheadOfTheServer(env, page, "/marketplace/agent/[creator]/[slug]"),
  "/marketplace/skills/[slug]": (env, page) =>
    visitAheadOfTheServer(env, page, "/marketplace/skills/[slug]"),
};

/** The Marketplace's sample for `route`, opened signed out, the browser's
 *  clock two days ahead of the server's. */
async function visitAheadOfTheServer(env: CrawlEnv, page: Page, route: string) {
  const { samples } = await sampleMarketplace(page);
  const sample = samples.find((candidate) => candidate.route === route);
  test.skip(!sample, `nothing on this stack to open at ${route}`);
  if (!sample) return [];
  return crawl(env, [
    planVisit(
      route,
      { who: "visitor" },
      { ...sample, clockAheadMs: 2 * DAY_MS },
    ),
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

function annotateUnsampled(unsampled: Unsampled[]) {
  for (const { route, why } of unsampled) {
    test.info().annotations.push({
      type: "unsampled route",
      description: `${route}: ${why}`,
    });
  }
}

async function expectAllClean(results: VisitResult[]) {
  await test.info().attach("pages.json", {
    body: JSON.stringify(summarize(results), null, 2),
    contentType: "application/json",
  });
  const failures = describeFailures(results);
  expect(
    failures.length,
    `${failures.length} of ${results.length} pages didn't load cleanly:\n\n` +
      failures.join("\n\n"),
  ).toBe(0);
}

// The pages in near-equal slices of at most PAGES_PER_TEST, in route order.
function slices(routes: string[]) {
  const count = Math.max(1, Math.ceil(routes.length / PAGES_PER_TEST));
  const size = Math.ceil(routes.length / count);
  return Array.from({ length: count }, (_, index) =>
    routes.slice(index * size, (index + 1) * size),
  ).filter((slice) => slice.length > 0);
}
