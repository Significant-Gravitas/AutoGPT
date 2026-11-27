import Constants from "expo-constants";
import PostHog from "posthog-react-native";

import { readConsent, saveConsent } from "./consent";
import {
  analyticsEnvironment,
  baseProperties,
  canSend,
  type AnalyticsEvent,
  type Consent,
} from "./rules";

const KEY = process.env.EXPO_PUBLIC_POSTHOG_KEY;
const HOST = process.env.EXPO_PUBLIC_POSTHOG_HOST || "https://eu.i.posthog.com";

let client: PostHog | null = null;
let consent: Consent = "unset";
let userID: string | null = null;
let currentScreen: string | null = null;
let lastScreen: string | null = null;

function isAllowed() {
  return canSend({ consent, hasKey: !!KEY, isDev: __DEV__ });
}

function superProperties() {
  return baseProperties(
    analyticsEnvironment(process.env.EXPO_PUBLIC_SENTRY_ENVIRONMENT),
    Constants.expoConfig?.version ?? null,
  );
}

// Made only once the user allows it: even a client that sends no events
// fetches its project's settings from PostHog when it starts.
function openClient() {
  if (!KEY || !isAllowed()) return;
  if (!client) {
    try {
      client = new PostHog(KEY, {
        host: HOST,
        defaultOptIn: false,
        captureAppLifecycleEvents: false,
        enableSessionReplay: false,
        errorTracking: { autocapture: false },
        preloadFeatureFlags: false,
        disableRemoteFeatureFlags: true,
        disableSurveys: true,
      });
      void client.register(superProperties());
    } catch {
      client = null;
    }
  }
  if (!client) return;
  void client.optIn();
  if (userID) client.identify(userID);
}

/** Whether this build has somewhere to send usage data, so asking about it means something. */
export function hasAnalyticsKey() {
  return !!KEY;
}

/** Usage analytics, in store and dev builds only, and only once the user allows it. */
export function startAnalytics() {
  consent = readConsent();
  openClient();
  sendCurrentScreen();
}

export function setAnalyticsConsent(allowed: boolean) {
  consent = allowed ? "allowed" : "declined";
  saveConsent(consent);
  // A screen sent before the change goes again once allowed.
  lastScreen = null;
  if (allowed) {
    openClient();
    sendCurrentScreen();
  } else {
    void client?.optOut();
  }
}

/** The platform user id only: the email and name never go to analytics. */
export function identifyUser(id: string) {
  userID = id;
  if (client && isAllowed()) client.identify(id);
}

export function resetAnalytics() {
  userID = null;
  currentScreen = null;
  lastScreen = null;
  if (!client) return;
  client.reset();
  void client.register(superProperties());
  // reset() forgets the opt-in too; the choice belongs to the phone, not the account.
  if (isAllowed()) void client.optIn();
}

export function track(event: AnalyticsEvent, properties?: Record<string, string>) {
  if (!client || !isAllowed()) return;
  client.capture(event, properties);
}

export function trackScreen(name: string) {
  currentScreen = name;
  sendCurrentScreen();
}

function sendCurrentScreen() {
  if (!client || !isAllowed() || currentScreen === null || currentScreen === lastScreen) return;
  lastScreen = currentScreen;
  void client.screen(currentScreen);
}
