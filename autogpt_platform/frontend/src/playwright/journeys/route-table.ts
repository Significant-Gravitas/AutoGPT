import fs from "fs";
import path from "path";
import type { ExpectedResponse } from "./guard";

// Which pages the route crawl visits, as whom, and what each one must show.
// The pages come from src/app at test time (listAppRoutes), so a page added
// later is crawled without an entry here: as a page for a signed-in user,
// or for an admin under /admin. A page needs an entry only when that default
// doesn't fit it, and every entry says why.

export type Access = "public" | "signed-in" | "admin";

/** Whom a page is opened as: signed out, a signed-in user, or an admin. */
export type Who = "visitor" | "user" | "admin";

/**
 * An open bug a page fails on. The page is left out of its group and crawled
 * in a test of its own per bug, pinned to it (its visit is in the spec's
 * REPRODUCTIONS when a plain one doesn't show the bug every time).
 */
export interface KnownBug {
  route: string;
  issue: number;
  /** Whom the bug shows for. */
  who: Who;
  /** What goes wrong, for the pin. */
  what: string;
  /** The problems the bug causes, as the crawl words them. Any other
   *  problem on the page still fails the test. */
  problem: RegExp;
  /** Other pages can show it too, depending on the data they list. Their
   *  groups don't fail on it: this bug's own test pins it. */
  showsElsewhere?: boolean;
}

export interface RouteRule {
  /** Why the page needs an entry. */
  reason: string;
  /** "public": for anyone, so it is crawled signed out and signed in. */
  access?: Access;
  /** What to open, when the bare route isn't what a person opens. */
  visit?: string;
  /** Where the visit ends up, when the page sends the person on by design. */
  landsOn?: string;
  /** Where a signed-in visit to a public page ends up. */
  signedInLandsOn?: string;
  /** Text the page shows once it has loaded. */
  shows?: string;
  /** Showing the error card is the page's job. */
  showsErrorCard?: boolean;
  /** API responses of 400 or more the page gets by design. */
  expectedResponses?: ExpectedResponse[];
  /** It ends the session it is opened with, so it gets one of its own. */
  ownSession?: boolean;
  /** Only a local stack serves it; anywhere else it is a 404. */
  localOnly?: boolean;
  /**
   * Only a stack that takes payments serves it: a hosted one, or a
   * self-hosted one whose operator turned payments on. On any other
   * self-hosted stack it is a 404.
   */
  hostedOnly?: boolean;
}

/**
 * An open bug that shows now and then on pages that are otherwise fine, with
 * no visit that shows it every time, so no test can pin it: CI accepts a
 * pinned test only if it fails every run. A group notes it on its pages with
 * an annotation, rather than fail on it.
 */
export interface IntermittentBug {
  issue: number;
  what: string;
  problem: RegExp;
}

export const INTERMITTENT_BUGS: IntermittentBug[] = [
  {
    issue: 16925,
    what: "a page throws React error #418 (HTML) while it hydrates, now and then",
    problem:
      /Minified React error #418; visit https:\/\/react\.dev\/errors\/418\?args\[\]=HTML/,
  },
];

export const KNOWN_BUGS: KnownBug[] = [];

export function hasKnownBug(route: string) {
  return KNOWN_BUGS.some((bug) => bug.route === route);
}

const MISSING_OAUTH_PARAMETERS = {
  showsErrorCard: true,
  shows: "Missing required parameters",
} as const;

export const ROUTE_RULES: Record<string, RouteRule> = {
  "/": {
    reason: "sends a signed-in user on to Home",
    landsOn: "/home",
  },
  "/account-closed": {
    reason:
      "is for a closed account; an open one goes back to the app, which opens Home",
    landsOn: "/home",
  },
  "/logout": {
    reason: "signs the user out, then opens the login page",
    landsOn: "/login",
    ownSession: true,
    // Requests the app's providers had in flight lose the race with the
    // sign-out and answer 401; the user never sees them.
    expectedResponses: [{ status: 401, url: /\/api\/proxy\/api\// }],
  },
  "/onboarding": {
    reason: "sends a user who finished onboarding on to Home",
    landsOn: "/home",
  },
  "/copilot": {
    reason: "AutoPilot moved to Home: an old link goes there, query and all",
    landsOn: "/home",
  },
  "/link/slack/installed": {
    reason:
      "hands a new Slack install to the Slack app, then goes back to bot settings",
    landsOn: "/settings/bots",
  },
  "/settings": {
    reason: "opens the first settings page",
    landsOn: "/settings/profile",
  },
  "/profile/api-keys": {
    reason: "an old link: the page moved to Settings",
    landsOn: "/settings/api-keys",
  },
  "/profile/credits": {
    reason: "an old link: the page moved to Settings, to Billing",
    landsOn: "/settings/billing",
    hostedOnly: true,
  },
  "/settings/billing": {
    reason:
      "Billing: a self-hosted stack without payments answers it with the not-found page (useBillingStatus)",
    hostedOnly: true,
  },
  "/profile/dashboard": {
    reason: "an old link: the page moved to Settings",
    landsOn: "/settings/creator-dashboard",
  },
  "/auth/authorize": {
    reason:
      "an OAuth app's consent page: opened without the app's request, it says what's missing",
    ...MISSING_OAUTH_PARAMETERS,
  },
  "/auth/integrations/setup-wizard": {
    reason:
      "an OAuth app's integrations setup: opened without the app's request, it says what's missing",
    ...MISSING_OAUTH_PARAMETERS,
  },
  "/admin/impersonation": {
    reason: "opens Users, where an admin starts impersonating",
    landsOn: "/admin/users",
  },
  "/admin/rate-limits": {
    reason: "rate limits moved into Users",
    landsOn: "/admin/users",
  },
  "/admin/test-data": {
    reason: "the backend's test-data routes exist only on a local stack",
    localOnly: true,
  },
  "/dev/brain-dump-debug": {
    reason:
      "a developer page: a 404 in production, a page for a signed-in user on this stack",
  },

  // For anyone, signed in or not.
  "/login": {
    reason: "the sign-in form; a signed-in user goes on to Home",
    access: "public",
    signedInLandsOn: "/home",
  },
  "/signup": {
    reason: "the sign-up form; a signed-in user goes on to Home",
    access: "public",
    signedInLandsOn: "/home",
  },
  "/reset-password": {
    reason: "asks for the email to send a reset link to",
    access: "public",
  },
  "/microsoft-setup": {
    reason: "setup notes for a Microsoft admin, who may have no account",
    access: "public",
    shows: "Set up Microsoft 365 for your organization",
  },
  "/marketplace": {
    reason: "the Marketplace is open to all",
    access: "public",
  },
  "/marketplace/search": {
    reason: "the Marketplace is open to all",
    access: "public",
  },
  "/marketplace/skills": {
    reason: "the Marketplace is open to all",
    access: "public",
  },
  "/tour": { reason: "the product tour, for prospects", access: "public" },
  "/tour/chat": { reason: "the product tour, for prospects", access: "public" },
  "/health": {
    reason: "the frontend's health check",
    access: "public",
    shows: "Yay im healthy",
  },
  "/unauthorized": {
    reason: "says the visitor may not see a page",
    access: "public",
    shows: "Unauthorized Access",
  },
  "/error": {
    reason: "the app's error page: the error card is what it shows",
    access: "public",
    showsErrorCard: true,
    shows: "An unexpected error occurred",
  },
  "/auth/auth-code-error": {
    reason:
      "shows the error a failed sign-in puts in its URL's hash; with none it waits for one",
    access: "public",
    visit:
      "/auth/auth-code-error#error=server_error&error_code=bad_oauth_callback&error_description=The+sign-in+could+not+be+finished",
    showsErrorCard: true,
    shows: "Error code: bad_oauth_callback",
  },
};

/** The route crawl's test that samples a page with a dynamic segment. */
export type Sampler = "marketplace" | "own data" | "shared" | "admin";

export interface DynamicRoute {
  /** The test that samples it. It fails unless it crawls the page or says
   *  why it can't, so an entry here always ends up crawled or noted. */
  sampledIn: Sampler;
  /** What the page is opened with. */
  sample: string;
}

/** How the crawl samples each page with a dynamic segment, and in which test. */
export const DYNAMIC_ROUTES: Record<string, DynamicRoute> = {
  "/marketplace/agent/[creator]/[slug]": {
    sampledIn: "marketplace",
    sample: "an approved listing on the stack (CI loads the official ones)",
  },
  "/marketplace/creator/[creator]": {
    sampledIn: "marketplace",
    sample: "the creator of that listing",
  },
  "/marketplace/experts/[expertId]": {
    sampledIn: "marketplace",
    sample: "an expert on the published roster",
  },
  "/marketplace/skills/[slug]": {
    sampledIn: "marketplace",
    sample: "a skill in the Marketplace",
  },
  "/library/agents/[id]": {
    sampledIn: "own data",
    sample: "a workflow the journey makes",
  },
  "/team/[expertId]": {
    sampledIn: "own data",
    sample: "an expert the journey hires",
  },
  "/link/[token]": {
    sampledIn: "own data",
    sample:
      "an unknown setup link, which shows the page's own failure state: only a chat bot makes real ones",
  },
  "/share/[token]": { sampledIn: "shared", sample: "a run the journey shares" },
  "/share/chat/[token]": {
    sampledIn: "shared",
    sample: "a chat the journey shares",
  },
  "/admin/marketplace/preview/[id]": {
    sampledIn: "admin",
    sample: "a listing the journey submits",
  },
};

/**
 * The routes `table` gives `sampler` that a test of it has no sample for
 * (`missing`), and the ones it samples that `table` gives another test or
 * none (`misplaced`). `sampled` counts a route it says it can't sample.
 */
export function samplingGaps(
  sampler: Sampler,
  sampled: string[],
  table: Record<string, DynamicRoute> = DYNAMIC_ROUTES,
) {
  return {
    missing: Object.keys(table).filter(
      (route) => table[route].sampledIn === sampler && !sampled.includes(route),
    ),
    misplaced: sampled.filter((route) => table[route]?.sampledIn !== sampler),
  };
}

const APP_DIR = path.resolve(__dirname, "../../app");
const PAGE_FILE = /^page\.(tsx|ts|jsx|js|mdx)$/;

/**
 * Every page in src/app, as the path Next.js serves it at: route groups
 * dropped, dynamic segments as written ("/library/agents/[id]").
 */
export function listAppRoutes(appDir = APP_DIR): string[] {
  const routes: string[] = [];
  collectPages(appDir, [], routes);
  // A slot's page and its parent's are served at one URL.
  return [...new Set(routes)].sort();
}

function collectPages(dir: string, segments: string[], routes: string[]) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isFile() && PAGE_FILE.test(entry.name)) {
      routes.push(`/${segments.join("/")}`);
    } else if (entry.isDirectory() && servesPages(entry.name)) {
      collectPages(
        path.join(dir, entry.name),
        addsSegment(entry.name) ? [...segments, entry.name] : segments,
        routes,
      );
    }
  }
}

// A private folder (_lib) or an intercepting route ((.)photo) serves no URL
// of its own.
function servesPages(folder: string) {
  return !folder.startsWith("_") && !/^\(\.{1,3}\)/.test(folder);
}

// A route group ((platform)) or a parallel route's slot (@modal) adds no
// segment to the URL.
function addsSegment(folder: string) {
  return !/^\(.+\)$/.test(folder) && !folder.startsWith("@");
}

export function isDynamic(route: string) {
  return route.includes("[");
}

export function accessOf(route: string): Access {
  const rule = ROUTE_RULES[route];
  if (rule?.access) return rule.access;
  return route === "/admin" || route.startsWith("/admin/")
    ? "admin"
    : "signed-in";
}
