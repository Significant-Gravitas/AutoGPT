import * as Sentry from "@sentry/nextjs";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { reportMismatchToSentry } from "../dual";

const sent: Sentry.ErrorEvent[] = [];

beforeAll(() => {
  vi.spyOn(console, "info").mockImplementation(() => undefined);
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    transport: () => ({
      send: () => Promise.resolve({}),
      flush: () => Promise.resolve(true),
    }),
    beforeSend(event) {
      sent.push(event);
      return null;
    },
  });
});

beforeEach(() => {
  sent.length = 0;
});

afterAll(() => {
  vi.restoreAllMocks();
});

describe("feature-flag mismatch report", () => {
  it("names the flag and which way the vendors disagree", async () => {
    reportMismatchToSentry({
      flag: "enable-platform-payment",
      launchdarkly: { value: true },
      posthog: { value: false },
    });
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    expect(sent[0].message).toBe(
      "Feature flag mismatch: enable-platform-payment (LaunchDarkly=true, PostHog=false)",
    );
    expect(sent[0].level).toBe("warning");
    expect(sent[0].tags).toMatchObject({
      "feature_flag.key": "enable-platform-payment",
      "feature_flag.launchdarkly": "true",
      "feature_flag.posthog": "false",
    });
  });

  it("files each flag as its own issue, whichever way it disagrees", async () => {
    reportMismatchToSentry({
      flag: "enable-platform-payment",
      launchdarkly: { value: true },
      posthog: { value: false },
    });
    reportMismatchToSentry({
      flag: "enable-platform-payment",
      launchdarkly: { value: false },
      posthog: { value: true },
    });
    reportMismatchToSentry({
      flag: "copilot-models",
      launchdarkly: { value: { default: "a" } },
      posthog: { value: null },
    });
    await Sentry.flush(2000);

    const [first, flipped, other] = sent;
    expect(first.fingerprint).toEqual([
      "feature-flag-mismatch",
      "enable-platform-payment",
    ]);
    expect(flipped.fingerprint).toEqual(first.fingerprint);
    expect(other.fingerprint).toEqual([
      "feature-flag-mismatch",
      "copilot-models",
    ]);
    expect(other.message).toBe(
      'Feature flag mismatch: copilot-models (LaunchDarkly={"default":"a"}, PostHog=null)',
    );
  });
});
