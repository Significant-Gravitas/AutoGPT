export type Consent = "unset" | "allowed" | "declined";

export type AnalyticsEnvironment = "prod" | "dev" | "local";

// Names from docs/engineering/tracking-plan.md, the same strings the web
// sends. Never rename one: PostHog keeps the raw string.
export const EVENTS = {
  homeViewed: "home_viewed",
  homeTeamMemberClicked: "home_team_member_clicked",
  expertProfileOpened: "expert_profile_opened",
  hireStarted: "hire_started",
  expertThreadCreated: "expert_thread_created",
} as const;

export type AnalyticsEvent = (typeof EVENTS)[keyof typeof EVENTS];

export function isConsent(value: unknown): value is Consent {
  return value === "unset" || value === "allowed" || value === "declined";
}

export function canSend({
  consent,
  hasKey,
  isDev,
}: {
  consent: Consent;
  hasKey: boolean;
  isDev: boolean;
}) {
  return consent === "allowed" && hasKey && !isDev;
}

export function analyticsEnvironment(sentryEnvironment: string | undefined): AnalyticsEnvironment {
  if (sentryEnvironment === "production") return "prod";
  if (sentryEnvironment === "dev") return "dev";
  return "local";
}

export function baseProperties(environment: AnalyticsEnvironment, appVersion: string | null) {
  return {
    environment,
    source: "mobile",
    ...(appVersion ? { app_version: appVersion } : {}),
  };
}

export function screenName(segments: string[]) {
  const parts = segments
    .flatMap((segment) => segment.split("/"))
    .filter((part) => part !== "" && part !== "index" && !/^\(.*\)$/.test(part));
  return parts.length > 0 ? parts.join("/") : "home";
}

interface NavigationState {
  index?: number;
  routes: { name: string; state?: NavigationState }[];
}

/** The route names down to the screen in front, e.g. ["(app)", "chat/[sessionId]"]. */
export function focusedRouteNames(state: NavigationState | undefined): string[] {
  if (!state || state.routes.length === 0) return [];
  const route = state.routes[state.index ?? state.routes.length - 1];
  if (!route) return [];
  return [route.name, ...focusedRouteNames(route.state)];
}
