import { randomBytes, randomInt } from "crypto";
import { expect, type Page } from "@playwright/test";
import { grantSubscriptionTier } from "./subscription-tier";

export interface JourneyUser {
  email: string;
  password: string;
  name: string;
  id?: string;
}

// Onboarding steps a journey user has already done. ONBOARDING_COMPLETE also
// grants the onboarding credits, which is how a fresh user gets a balance
// when credits are on; the rest dismiss first-visit cards that would cover
// the page.
const COMPLETED_ONBOARDING_STEPS = [
  "ONBOARDING_COMPLETE",
  "AGENT_NEW_RUN",
  "CAPABILITY_CARDS",
  "FIRST_RUN_TOUR",
  "CHATS_TIP",
  "COST_TIP",
  "EXPERT_TIP",
  "AGENTS_TAB_INTRO",
  "MARKETPLACE_TAB_INTRO",
  "BUILD_TAB_INTRO",
  "WORKFLOWS_MOVED",
];

export interface ProvisionOptions {
  /**
   * Whether the user gets PLAYWRIGHT_JOURNEY_TIER's plan (`subscription-tier.ts`),
   * as most journeys need to get past the hosted paywall. A journey about
   * buying a plan passes false: its user keeps the tier a new hosted user
   * really has.
   */
  grantTier?: boolean;
}

/**
 * Signs a user into the page's browser context, ready to use the app.
 *
 * By default this signs up a brand-new user, so a journey never shares state
 * with another one and never depends on CI-only seed data. Set
 * PLAYWRIGHT_JOURNEY_EMAIL and PLAYWRIGHT_JOURNEY_PASSWORD to sign in as an
 * existing account instead, e.g. a paid user on a preview where fresh users
 * meet the paywall.
 */
export async function provisionJourneyUser(
  page: Page,
  options: ProvisionOptions = {},
): Promise<JourneyUser> {
  const existingEmail = process.env.PLAYWRIGHT_JOURNEY_EMAIL;
  const existingPassword = process.env.PLAYWRIGHT_JOURNEY_PASSWORD;
  if (!existingEmail || !existingPassword) {
    return provisionNewJourneyUser(page, options);
  }
  const user = await signIn(page, {
    email: existingEmail,
    password: existingPassword,
    name: existingEmail.split("@")[0],
  });
  return readyForTheApp(page, user, options);
}

/**
 * Signs a brand-new user into the page's browser context, ready to use the
 * app, even when PLAYWRIGHT_JOURNEY_EMAIL names an account: for the other
 * people in a journey (an admin, a second user), who must never turn out to
 * be its first user.
 */
export async function provisionNewJourneyUser(
  page: Page,
  options: ProvisionOptions = {},
): Promise<JourneyUser> {
  return readyForTheApp(page, await signUp(page, newJourneyUser()), options);
}

async function readyForTheApp(
  page: Page,
  user: JourneyUser,
  { grantTier = true }: ProvisionOptions,
) {
  // Before anything reads the tier: the backend caches it for minutes.
  if (user.id && grantTier) await grantSubscriptionTier(user.id);
  await finishAccountSetup(page);
  await completeOnboarding(page);
  return user;
}

/**
 * Signs up a brand-new user who has finished onboarding but not the account
 * setup (POST /auth/user) that the app's sign-up runs before the first page.
 * A Google or email-link sign-in is in this state until /auth/callback has
 * run, and stays in it if that redirect is lost.
 */
export async function signUpBeforeAccountSetup(
  page: Page,
): Promise<JourneyUser> {
  const user = await signUp(page, newJourneyUser());
  if (user.id) await grantSubscriptionTier(user.id);
  await completeOnboarding(page);
  return user;
}

/**
 * Creates a brand-new account that has not started onboarding, then signs the
 * browser out again, so the journey can log in through the app's own form.
 */
export async function signUpWithoutOnboarding(
  page: Page,
): Promise<JourneyUser> {
  const user = await signUp(page, newJourneyUser());
  await page.context().clearCookies();
  return user;
}

export function newJourneyUser(): JourneyUser {
  const id = `${Date.now()}-${randomBytes(3).toString("hex")}`;
  return {
    email: `journey-${id}@${journeyEmailDomain()}`,
    password: `Journey-${randomBytes(12).toString("hex")}`,
    name: `Journey ${id}`,
  };
}

// Previews only accept sign-ups from allowlisted domains; @previews.agpt.co
// keeps password sign-up and needs no inbox.
function journeyEmailDomain() {
  if (process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN) {
    return process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN;
  }
  return isLoopback(appOrigin()) ? "example.com" : "previews.agpt.co";
}

async function signUp(page: Page, user: JourneyUser) {
  const response = await page.request.post("/api/auth/sign-up/email", {
    headers: authHeaders(),
    data: user,
  });
  expect(
    response.ok(),
    `sign-up for ${user.email} failed: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  const body = (await response.json()) as { user?: { id?: string } };
  return { ...user, id: body.user?.id };
}

/**
 * Signs an existing user in, giving the page's context a new session. Sign
 * in again after changing the user's row: a session caches the user it
 * started with for five minutes.
 */
export async function signIn(page: Page, user: JourneyUser) {
  const response = await page.request.post("/api/auth/sign-in/email", {
    headers: authHeaders(),
    data: { email: user.email, password: user.password },
  });
  expect(
    response.ok(),
    `sign-in for ${user.email} failed: ${response.status()}`,
  ).toBe(true);
  const body = (await response.json()) as { user?: { id?: string } };
  return { ...user, id: body.user?.id };
}

// The app's sign-up, sign-in and /auth/callback do this before the user's
// first page; among other things it creates their personal org.
async function finishAccountSetup(page: Page) {
  const response = await page.request.post("/api/proxy/api/auth/user");
  expect(
    response.ok(),
    `account setup failed: ${response.status()} ${await response.text()}`,
  ).toBe(true);
}

async function completeOnboarding(page: Page) {
  for (const step of COMPLETED_ONBOARDING_STEPS) {
    const response = await page.request.post("/api/proxy/api/onboarding/step", {
      params: { step },
    });
    expect(
      response.ok(),
      `onboarding step ${step} failed: ${response.status()} ${await response.text()}`,
    ).toBe(true);
  }
}

// The app caps sign-ups per client IP. On a local or CI stack every journey
// comes from one address, so each sign-up claims its own; a deployed app
// behind a proxy sets the real address and the cap applies as usual.
export function authHeaders(): Record<string, string> {
  const origin = appOrigin();
  if (!isLoopback(origin)) return { Origin: origin };
  return {
    Origin: origin,
    "X-Forwarded-For": `10.${randomInt(256)}.${randomInt(256)}.${randomInt(1, 255)}`,
  };
}

/** Whether the journeys run against a stack on this machine, not a deployed one. */
export function isLocalStack() {
  return isLoopback(appOrigin());
}

function appOrigin() {
  return new URL(process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000")
    .origin;
}

function isLoopback(origin: string) {
  const { hostname } = new URL(origin);
  return ["localhost", "127.0.0.1", "[::1]"].includes(hostname);
}
