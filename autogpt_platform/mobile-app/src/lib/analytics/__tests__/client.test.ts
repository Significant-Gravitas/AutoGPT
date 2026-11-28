import type * as ClientModule from "../client";
import type { Consent } from "../rules";

const mockReadConsent = jest.fn<Consent, []>();
const mockSaveConsent = jest.fn<boolean, ["allowed" | "declined"]>();
const mockClient = {
  register: jest.fn(() => Promise.resolve()),
  optIn: jest.fn(() => Promise.resolve()),
  optOut: jest.fn(() => Promise.resolve()),
  identify: jest.fn(),
  capture: jest.fn(),
  screen: jest.fn(() => Promise.resolve()),
  reset: jest.fn(),
};
const mockPostHog = jest.fn(() => mockClient);

jest.mock("posthog-react-native", () => ({ __esModule: true, default: mockPostHog }));
jest.mock("expo-constants", () => ({ expoConfig: { version: "0.2.0" } }));
jest.mock("../consent", () => ({
  readConsent: () => mockReadConsent(),
  saveConsent: (consent: "allowed" | "declined") => mockSaveConsent(consent),
}));

const globals = globalThis as { __DEV__?: boolean };
const wasDev = globals.__DEV__;

// A fresh copy per test, so the key and the module's own state start over.
function loadClient(consent: Consent) {
  mockReadConsent.mockReturnValue(consent);
  let client: typeof ClientModule | undefined;
  jest.isolateModules(() => {
    client = jest.requireActual<typeof ClientModule>("../client");
  });
  if (!client) throw new Error("client did not load");
  return client;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockSaveConsent.mockReturnValue(true);
  process.env.EXPO_PUBLIC_POSTHOG_KEY = "test-key";
  globals.__DEV__ = false;
});

afterAll(() => {
  delete process.env.EXPO_PUBLIC_POSTHOG_KEY;
  globals.__DEV__ = wasDev;
});

test("makes no client before the user answers", () => {
  loadClient("unset").startAnalytics();
  expect(mockPostHog).not.toHaveBeenCalled();
});

test("makes no client after Not now", () => {
  loadClient("declined").startAnalytics();
  expect(mockPostHog).not.toHaveBeenCalled();
});

test("makes one client and opts in once allowed", () => {
  loadClient("allowed").startAnalytics();
  expect(mockPostHog).toHaveBeenCalledTimes(1);
  expect(mockClient.optIn).toHaveBeenCalled();
});

test("nothing reaches PostHog before the user answers", () => {
  const client = loadClient("unset");
  client.startAnalytics();
  client.identifyUser("user-1");
  client.track("home_viewed");
  client.trackScreen("home");
  expect(mockPostHog).not.toHaveBeenCalled();
});

test("allowing later makes the client and opts in", () => {
  const client = loadClient("unset");
  client.startAnalytics();
  client.identifyUser("user-1");
  client.trackScreen("home");
  client.setAnalyticsConsent(true);
  expect(mockPostHog).toHaveBeenCalledTimes(1);
  expect(mockClient.optIn).toHaveBeenCalled();
  expect(mockClient.identify).toHaveBeenCalledWith("user-1");
  expect(mockClient.screen).toHaveBeenCalledWith("home");
});

test("allowing again sends the screen in front again", () => {
  const client = loadClient("allowed");
  client.startAnalytics();
  client.trackScreen("settings");
  client.setAnalyticsConsent(false);
  client.trackScreen("plan");
  expect(mockClient.screen).toHaveBeenCalledTimes(1);
  client.trackScreen("settings");
  client.setAnalyticsConsent(true);
  client.trackScreen("settings");
  expect(mockClient.optOut).toHaveBeenCalled();
  expect(mockClient.screen.mock.calls).toEqual([["settings"], ["settings"]]);
});

test("a failed revocation stops tracking until the user can save it", () => {
  const client = loadClient("allowed");
  client.startAnalytics();
  mockSaveConsent.mockReturnValueOnce(false);
  expect(client.setAnalyticsConsent(false)).toBe(false);
  expect(client.analyticsConsentState()).toEqual({ sharesUsage: false, unsavedChoice: false });
  client.startAnalytics();
  client.track("home_viewed");
  expect(mockClient.capture).not.toHaveBeenCalled();
  expect(mockClient.optOut).toHaveBeenCalled();

  expect(client.setAnalyticsConsent(false)).toBe(true);
  expect(client.analyticsConsentState()).toEqual({ sharesUsage: false, unsavedChoice: null });
});

test("an opt-in that could not be saved does not start tracking", () => {
  const client = loadClient("declined");
  client.startAnalytics();
  mockSaveConsent.mockReturnValueOnce(false);
  expect(client.setAnalyticsConsent(true)).toBe(false);
  expect(mockPostHog).not.toHaveBeenCalled();
});
