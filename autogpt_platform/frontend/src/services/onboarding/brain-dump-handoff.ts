// Handoff between the onboarding wizard and the copilot home.
//
// The wizard records which path the user took; the copilot home consumes
// it exactly once and asks the server for the matching intro. Only the
// path letter travels — the transcript stays on the server, which is the
// point of composing the prompt there.

const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";
const MIC_GLOW_KEY = "autogpt:onboarding-mic-glow";
const AWAITING_FOLLOWUP_KEY = "autogpt:onboarding-intro-awaiting-followup";
const PENDING_LATER_DUMP_KEY = "autogpt:onboarding-pending-later-dump";

export type IntroPath = "A" | "B";

// Path B ends with AutoPilot asking for a dump, so the mic it points at
// gets a one-time highlight.
export function setMicGlow() {
  setFlag(MIC_GLOW_KEY);
}

export function takeMicGlow() {
  return takeFlag(MIC_GLOW_KEY);
}

// Set when AutoPilot's intro goes out, consumed by the user's first real
// message afterwards. Measures whether the intro actually started a
// conversation.
export function setIntroAwaitingFollowup() {
  setFlag(AWAITING_FOLLOWUP_KEY);
}

export function takeIntroAwaitingFollowup() {
  return takeFlag(AWAITING_FOLLOWUP_KEY);
}

// Path B only: the user skipped the dump, so AutoPilot invited them to
// record one from the copilot composer instead. Consumed the first time
// they finish a voice message there.
export function setPendingLaterDump() {
  setFlag(PENDING_LATER_DUMP_KEY);
}

export function takePendingLaterDump() {
  return takeFlag(PENDING_LATER_DUMP_KEY);
}

// Set when the user leaves onboarding for the copilot home, cleared when
// they close the full-screen welcome overlay there. Refresh-safe: the
// overlay must keep covering the page (and keep the greeting unfetched)
// until it is deliberately dismissed.
const WELCOME_PENDING_KEY = "autogpt:onboarding-welcome-pending";

export function setWelcomePending() {
  setFlag(WELCOME_PENDING_KEY);
}

export function peekWelcomePending() {
  if (typeof window === "undefined") return false;
  return window.sessionStorage.getItem(WELCOME_PENDING_KEY) === "1";
}

export function clearWelcomePending() {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(WELCOME_PENDING_KEY);
}

// The capability-cards first-run modal was completed or skipped. The
// backend's CAPABILITY_CARDS onboarding step is the source of truth;
// this is its local cache so the modal never reshows even if that call
// failed or hasn't been checked. Value is the user id, same reasoning
// as the greeting flag below.
const CAPABILITY_CARDS_KEY = "autogpt:copilot-capability-cards-seen";

export function peekCapabilityCardsSeen(userId: string | null | undefined) {
  if (typeof window === "undefined" || !userId) return false;
  return window.localStorage.getItem(CAPABILITY_CARDS_KEY) === userId;
}

export function setCapabilityCardsSeen(userId: string | null | undefined) {
  if (typeof window === "undefined" || !userId) return;
  window.localStorage.setItem(CAPABILITY_CARDS_KEY, userId);
}

// The greeting is retired for good the first time the user sends a
// message. localStorage rather than sessionStorage because it must
// survive new tabs and restarts; the backend flag is the source of
// truth and this is only its cache — absence means "ask the server".
// The stored value is the user id, not a boolean: a different account
// signing in on the same browser must not inherit the previous
// account's "done" and silently skip its greeting.
const GREETING_DONE_KEY = "autogpt:copilot-greeting-done";

export function peekGreetingDone(userId: string | null | undefined) {
  if (typeof window === "undefined" || !userId) return false;
  return window.localStorage.getItem(GREETING_DONE_KEY) === userId;
}

export function setGreetingDone(userId: string | null | undefined) {
  if (typeof window === "undefined" || !userId) return;
  window.localStorage.setItem(GREETING_DONE_KEY, userId);
}

function setFlag(key: string) {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(key, "1");
}

function takeFlag(key: string) {
  if (typeof window === "undefined") return false;
  const value = window.sessionStorage.getItem(key) === "1";
  if (value) window.sessionStorage.removeItem(key);
  return value;
}

export function setIntroPath(path: IntroPath) {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(INTRO_PATH_KEY, path);
}

export function peekIntroPath(): IntroPath | null {
  if (typeof window === "undefined") return null;
  const raw = window.sessionStorage.getItem(INTRO_PATH_KEY);
  return raw === "A" || raw === "B" ? raw : null;
}

// Read-and-clear: the intro is a one-time event, and a refresh of the
// copilot home must not replay it.
export function takeIntroPath(): IntroPath | null {
  const path = peekIntroPath();
  if (path && typeof window !== "undefined") {
    window.sessionStorage.removeItem(INTRO_PATH_KEY);
  }
  return path;
}
