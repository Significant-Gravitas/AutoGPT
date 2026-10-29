import {
  deleteCookie,
  readCookieNames,
  removeStorageKeys,
} from "./browser-storage";
import {
  getConsentAnswer,
  getConsentAnswerIdentity,
  isConsentAnswerUnconfirmed,
  subscribeToConsentAnswerPending,
  type ConsentCategory,
  type ConsentState,
} from "./consent";

// The stored answer whose leftovers were last deleted.
const CLEANED_ANSWER_KEY = "agpt_consent_cleanup";

interface Tracker {
  category: ConsentCategory;
  matches: (name: string) => boolean;
}

// What the third-party scripts behind each category leave in cookies and web
// storage: the Google tag's GA4 and Google Ads entries (`_gcl_ls` among them),
// and DataFast's visitor, session and page-view state. PostHog's storage is
// handled in providers/posthog/posthog-consent.ts.
const TRACKERS: readonly Tracker[] = [
  {
    category: "analytics",
    matches: (name) =>
      name === "_ga" || name.startsWith("_ga_") || name.startsWith("datafast_"),
  },
  {
    category: "advertising",
    matches: (name) => name.startsWith("_gcl_"),
  },
];

/**
 * Deletes what the trackers of every category the visitor's answer lacks
 * left in the browser, once per answer. The cookies live on `.agpt.co`, shared
 * with agpt.co, where Cookiebot keeps a separate answer: deleting on every
 * page load would keep taking what the visitor allowed there. So an answer
 * given on this page always cleans, and one found on load cleans only if it
 * has not already. Nothing goes while the answer is pending or only the
 * stored cookie's until uc.js confirms it, nor without a consent manager.
 * Returns the unsubscribe.
 */
export function followConsentForTrackerCookies(): () => void {
  let observed = false;
  let previous: string | null = null;

  function sync() {
    if (isConsentAnswerUnconfirmed()) return;
    const answer = getConsentAnswer();
    const identity = getConsentAnswerIdentity();
    const current = answer ? describeAnswer(answer, identity) : null;
    const answeredOnThisPage = observed && current !== previous;
    observed = true;
    previous = current;
    if (!answer) return;

    const declined = TRACKERS.filter(({ category }) => !answer[category]);
    if (declined.length === 0) return;
    if (!answeredOnThisPage && identity && readCleanedAnswer() === identity) {
      return;
    }
    forgetTrackers(declined);
    if (identity) rememberCleanedAnswer(identity);
  }

  sync();
  return subscribeToConsentAnswerPending(sync);
}

function describeAnswer(answer: ConsentState, identity: string | null) {
  return `${identity}|${answer.analytics}|${answer.advertising}`;
}

function forgetTrackers(declined: Tracker[]) {
  const matches = (name: string) =>
    declined.some((tracker) => tracker.matches(name));
  readCookieNames().filter(matches).forEach(deleteCookie);
  removeStorageKeys(() => window.localStorage, matches);
  removeStorageKeys(() => window.sessionStorage, matches);
}

function readCleanedAnswer(): string | null {
  try {
    return window.localStorage.getItem(CLEANED_ANSWER_KEY);
  } catch {
    return null;
  }
}

function rememberCleanedAnswer(identity: string) {
  try {
    window.localStorage.setItem(CLEANED_ANSWER_KEY, identity);
  } catch {
    // Storage blocked: the next page load cleans again.
  }
}
