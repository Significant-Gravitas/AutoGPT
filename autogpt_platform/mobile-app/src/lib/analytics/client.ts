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
  if (!client && KEY && !__DEV__) {
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
  if (!client || !isAllowed()) return;
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
}

export function setAnalyticsConsent(allowed: boolean) {
  consent = allowed ? "allowed" : "declined";
  saveConsent(consent);
  if (allowed) openClient();
  else void client?.optOut();
}

/** The platform user id only: the email and name never go to analytics. */
export function identifyUser(id: string) {
  userID = id;
  if (client && isAllowed()) client.identify(id);
}

export function resetAnalytics() {
  userID = null;
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
  if (!client || !isAllowed() || name === lastScreen) return;
  lastScreen = name;
  void client.screen(name);
}
