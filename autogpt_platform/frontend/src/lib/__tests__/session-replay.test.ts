import * as Sentry from "@sentry/nextjs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { setupSessionReplay } from "../session-replay";

vi.mock("@sentry/nextjs", () => ({
  addIntegration: vi.fn(),
  replayIntegration: vi.fn(() => ({ name: "Replay" })),
  replayCanvasIntegration: vi.fn(() => ({ name: "ReplayCanvas" })),
}));

function integrationNames(integrations: { name: string }[]) {
  return integrations.map((integration) => integration.name);
}

function addedIntegrationNames() {
  return vi
    .mocked(Sentry.addIntegration)
    .mock.calls.map(([integration]) => integration.name);
}

describe("setupSessionReplay", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    configureCookiebot();
  });

  afterEach(() => {
    removeCookiebot();
    vi.unstubAllEnvs();
  });

  it("installs replay at init when monitoring consent is already stored", () => {
    installCookiebot({ statistics: true });

    expect(integrationNames(setupSessionReplay())).toEqual([
      "ReplayCanvas",
      "Replay",
    ]);
  });

  it("records nothing before consent, then adds replay once it is granted", () => {
    installCookiebot();

    expect(setupSessionReplay()).toEqual([]);
    expect(Sentry.addIntegration).not.toHaveBeenCalled();

    answerCookiebot({ marketing: true });
    expect(Sentry.addIntegration).not.toHaveBeenCalled();

    answerCookiebot({ statistics: true });
    expect(addedIntegrationNames()).toEqual(["ReplayCanvas", "Replay"]);

    answerCookiebot({ statistics: true, marketing: true });
    expect(Sentry.addIntegration).toHaveBeenCalledTimes(2);
  });

  it("never installs replay without a consent banner", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    installCookiebot({ statistics: true });

    expect(setupSessionReplay()).toEqual([]);
    answerCookiebot({ statistics: true });
    expect(Sentry.addIntegration).not.toHaveBeenCalled();
  });
});
