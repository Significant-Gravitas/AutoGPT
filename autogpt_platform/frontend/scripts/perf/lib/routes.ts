import type { Page, Route } from "@playwright/test";

/**
 * Every URL the harness intercepts, in match order. Anything not listed goes
 * to the real dev server and backend untouched.
 *
 * Generated API hooks reach the backend through the Next proxy
 * (`/api/proxy/api/...`); the copilot stream goes straight to the backend
 * (`http://localhost:8006/api/chat/sessions/{id}/stream`), so that pattern
 * matches either host.
 */
const ROUTE_TABLE = [
  ["featureFlags", /^\/api\/proxy\/api\/feature-flags$/],
  ["onboardingCompleted", /^\/api\/proxy\/api\/onboarding\/completed$/],
  ["chatStream", /\/api\/chat\/sessions\/([^/]+)\/stream$/],
  ["chatSessionList", /^\/api\/proxy\/api\/chat\/sessions$/],
  ["chatSession", /^\/api\/proxy\/api\/chat\/sessions\/([^/]+)$/],
  ["chatSessionChild", /^\/api\/proxy\/api\/chat\/sessions\/([^/]+)\/(.+)$/],
  [
    "graphExecution",
    /^\/api\/proxy\/api\/graphs\/([^/]+)\/executions\/([^/]+)$/,
  ],
  ["graph", /^\/api\/proxy\/api\/graphs\/([^/]+)$/],
  ["libraryAgents", /^\/api\/proxy\/api\/library\/agents$/],
  ["executions", /^\/api\/proxy\/api\/executions$/],
  ["libraryAgent", /^\/api\/proxy\/api\/library\/agents\/([^/]+)$/],
  ["graphExecutions", /^\/api\/proxy\/api\/graphs\/([^/]+)\/executions$/],
  ["graphSchedules", /^\/api\/proxy\/api\/graphs\/([^/]+)\/schedules$/],
  ["libraryPresets", /^\/api\/proxy\/api\/library\/presets$/],
] as const;

export type RouteName = (typeof ROUTE_TABLE)[number][0];

export type RouteHandler = (
  route: Route,
  match: RegExpMatchArray,
) => Promise<void>;

export type RouteHandlers = Partial<Record<RouteName, RouteHandler>>;

export function matchRoute(url: string, handlers: RouteHandlers) {
  const { pathname } = new URL(url);
  for (const [name, pattern] of ROUTE_TABLE) {
    const handler = handlers[name];
    const match = handler ? pathname.match(pattern) : null;
    if (handler && match) return { name, handler, match };
  }
  return null;
}

export async function installRoutes(page: Page, handlers: RouteHandlers) {
  await page.route(
    (url) => matchRoute(url.toString(), handlers) !== null,
    async (route) => {
      const hit = matchRoute(route.request().url(), handlers);
      if (!hit) return route.fallback();
      await hit.handler(route, hit.match);
    },
  );
}

export function fulfillJSON(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

/**
 * Pins the parts of every page that would otherwise differ between machines:
 * feature flags fall back to their code defaults and onboarding counts as done.
 */
export function commonHandlers(): RouteHandlers {
  return {
    featureFlags: (route) =>
      fulfillJSON(route, { flags: {}, configured: false }),
    onboardingCompleted: (route) => fulfillJSON(route, { is_completed: true }),
  };
}
