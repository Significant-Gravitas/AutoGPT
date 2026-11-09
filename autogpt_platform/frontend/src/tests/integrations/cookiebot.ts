import { vi } from "vitest";

// Stands in for Cookiebot's uc.js: a configured domain group, the
// window.Cookiebot API and the events it fires when the visitor answers.

interface Answer {
  statistics?: boolean;
  marketing?: boolean;
  preferences?: boolean;
}

export const TEST_COOKIEBOT_CBID = "00000000-0000-4000-8000-000000000000";

export function configureCookiebot() {
  vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", TEST_COOKIEBOT_CBID);
}

export function installCookiebot(answer?: Answer) {
  const renew = vi.fn();
  window.Cookiebot = {
    consent: consentFrom(answer ?? {}),
    consented: false,
    declined: false,
    hasResponse: false,
    renew,
  };
  if (answer) setCookiebotAnswer(answer);
  return { renew };
}

/** The visitor answers the banner: updates the API, then fires the event. */
export function answerCookiebot(answer: Answer) {
  setCookiebotAnswer(answer);
  const accepted = Boolean(answer.statistics || answer.marketing);
  window.dispatchEvent(
    new Event(accepted ? "CookiebotOnAccept" : "CookiebotOnDecline"),
  );
}

export function removeCookiebot() {
  delete window.Cookiebot;
  document.cookie = "CookieConsent=; Path=/; Max-Age=0";
}

function setCookiebotAnswer(answer: Answer) {
  const cookiebot = window.Cookiebot;
  if (!cookiebot) throw new Error("installCookiebot() first");
  const consent = consentFrom(answer);
  const accepted = consent.statistics || consent.marketing;
  cookiebot.consent = consent;
  cookiebot.hasResponse = true;
  cookiebot.consented = accepted;
  cookiebot.declined = !accepted;
}

function consentFrom(answer: Answer) {
  return {
    necessary: true,
    preferences: answer.preferences ?? false,
    statistics: answer.statistics ?? false,
    marketing: answer.marketing ?? false,
  };
}
