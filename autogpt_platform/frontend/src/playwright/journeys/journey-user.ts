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
  "AGENTS_TAB_INTRO",
  "MARKETPLACE_TAB_INTRO",
  "BUILD_TAB_INTRO",
  "WORKFLOWS_MOVED",
];

/**
 * Signs a user into the page's browser context, ready to use the app.
 *
 * By default this signs up a brand-new user, so a journey never shares state
 * with another one and never depends on CI-only seed data. Set
 * PLAYWRIGHT_JOURNEY_EMAIL and PLAYWRIGHT_JOURNEY_PASSWORD to sign in as an
 * existing account instead, e.g. a paid user on a preview where fresh users
 * meet the paywall.
 */
export async function provisionJourneyUser(page: Page): Promise<JourneyUser> {
  const existingEmail = process.env.PLAYWRIGHT_JOURNEY_EMAIL;
  const existingPassword = process.env.PLAYWRIGHT_JOURNEY_PASSWORD;
  const user =
    existingEmail && existingPassword
      ? await signIn(page, {
          email: existingEmail,
          password: existingPassword,
          name: existingEmail.split("@")[0],
        })
      : await signUp(page, newJourneyUser());
  // Before anything reads the tier: the backend caches it for minutes.
  if (user.id) await grantSubscriptionTier(user.id);
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

function newJourneyUser(): JourneyUser {
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

async function signIn(page: Page, user: JourneyUser) {
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
function authHeaders(): Record<string, string> {
  const origin = appOrigin();
  if (!isLoopback(origin)) return { Origin: origin };
  return {
    Origin: origin,
    "X-Forwarded-For": `10.${randomInt(256)}.${randomInt(256)}.${randomInt(1, 255)}`,
  };
}

function appOrigin() {
  return new URL(process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000")
    .origin;
}

function isLoopback(origin: string) {
  const { hostname } = new URL(origin);
  return ["localhost", "127.0.0.1", "[::1]"].includes(hostname);
}
