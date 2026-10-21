import * as Sentry from "@sentry/nextjs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { redactRecordingEventURLs } from "../sentry-url-redaction";
import { setupSessionReplay } from "../session-replay";

vi.mock("@sentry/nextjs", () => ({
  addIntegration: vi.fn(),
  getReplay: vi.fn(),
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
      "Replay",
      "ReplayEventURLRedaction",
    ]);
  });

  it("records nothing before consent, then adds replay once it is granted", () => {
    installCookiebot();

    expect(setupSessionReplay()).toEqual([]);
    expect(Sentry.addIntegration).not.toHaveBeenCalled();

    answerCookiebot({ marketing: true });
    expect(Sentry.addIntegration).not.toHaveBeenCalled();

    answerCookiebot({ statistics: true });
    expect(addedIntegrationNames()).toEqual([
      "Replay",
      "ReplayEventURLRedaction",
    ]);

    answerCookiebot({ statistics: true, marketing: true });
    expect(Sentry.addIntegration).toHaveBeenCalledTimes(2);
  });

  it("masks all text, inputs and media unless an element opts in", () => {
    installCookiebot({ statistics: true });
    setupSessionReplay();

    expect(Sentry.replayIntegration).toHaveBeenCalledExactlyOnceWith({
      maskAllText: true,
      maskAllInputs: true,
      blockAllMedia: true,
      unmask: [".sentry-unmask, [data-sentry-unmask]"],
      beforeAddRecordingEvent: redactRecordingEventURLs,
    });
  });

  // Canvas pixels are recorded as drawn, with no masking at all.
  it("records no canvas", () => {
    installCookiebot({ statistics: true });
    setupSessionReplay();

    expect(Sentry.replayCanvasIntegration).not.toHaveBeenCalled();
  });

  it("never installs replay without a consent banner", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    installCookiebot({ statistics: true });

    expect(setupSessionReplay()).toEqual([]);
    answerCookiebot({ statistics: true });
    expect(Sentry.addIntegration).not.toHaveBeenCalled();
  });
});

describe("session replay on a page whose address carries a credential", () => {
  // The replay holds state across navigations, so each test loads it afresh.
  async function load() {
    vi.resetModules();
    return {
      Sentry: vi.mocked(await import("@sentry/nextjs")),
      ...(await import("../session-replay")),
    };
  }

  // A resume waits for the stop before it, so it lands on a later tick.
  async function settled() {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  function recordingReplay(mode: "session" | "buffer") {
    return {
      getReplayId: vi.fn(() => "r-1"),
      getRecordingMode: vi.fn(() => mode),
      stop: vi.fn(async () => undefined),
      start: vi.fn(),
      startBuffering: vi.fn(),
    };
  }

  beforeEach(() => {
    vi.clearAllMocks();
    configureCookiebot();
    installCookiebot({ statistics: true });
  });

  afterEach(() => {
    history.replaceState({}, "", "/");
    removeCookiebot();
    vi.unstubAllEnvs();
  });

  it.each(["/reset-password?token=r-1", "/link/t-1"])(
    "starts no replay on %s",
    async (page) => {
      history.replaceState({}, "", page);
      const { Sentry, setupSessionReplay } = await load();

      expect(integrationNames(setupSessionReplay())).toEqual([
        "ReplayEventURLRedaction",
      ]);
      expect(Sentry.replayIntegration).not.toHaveBeenCalled();
    },
  );

  it("starts one on an ordinary page, query and all", async () => {
    history.replaceState({}, "", "/copilot?sessionId=s-1");
    const { setupSessionReplay } = await load();

    expect(integrationNames(setupSessionReplay())).toEqual([
      "Replay",
      "ReplayEventURLRedaction",
    ]);
  });

  it("starts the replay once the visitor leaves the page", async () => {
    history.replaceState({}, "", "/link/t-1");
    const { Sentry, setupSessionReplay, syncReplayWithPage } = await load();
    setupSessionReplay();

    syncReplayWithPage(`${location.origin}/link/t-1`);
    await settled();
    expect(Sentry.addIntegration).not.toHaveBeenCalled();

    history.replaceState({}, "", "/library");
    syncReplayWithPage(location.href);
    await settled();
    expect(addedIntegrationNames()).toEqual(["Replay"]);
  });

  it("starts nothing when consent is granted there, until the visitor leaves", async () => {
    installCookiebot();
    history.replaceState({}, "", "/reset-password?token=r-1");
    const { setupSessionReplay, syncReplayWithPage } = await load();

    expect(setupSessionReplay()).toEqual([]);
    answerCookiebot({ statistics: true });
    expect(addedIntegrationNames()).not.toContain("Replay");

    history.replaceState({}, "", "/library");
    syncReplayWithPage(location.href);
    await settled();
    expect(
      addedIntegrationNames().filter((name) => name === "Replay"),
    ).toHaveLength(1);
  });

  it.each([
    ["session", "start"],
    ["buffer", "startBuffering"],
  ] as const)(
    "stops a %s recording on the way in and resumes it on the way out",
    async (mode, resume) => {
      const { Sentry, holdReplayBackFor, syncReplayWithPage } = await load();
      const replay = recordingReplay(mode);
      Sentry.getReplay.mockReturnValue(replay as never);

      // As Next calls it when the navigation starts, then once it has landed.
      holdReplayBackFor("/link/t-1");
      syncReplayWithPage(`${location.origin}/link/t-1`);
      await settled();
      expect(replay.stop).toHaveBeenCalledTimes(1);
      expect(replay.start).not.toHaveBeenCalled();
      expect(replay.startBuffering).not.toHaveBeenCalled();

      holdReplayBackFor("/library");
      syncReplayWithPage(`${location.origin}/library`);
      await settled();
      expect(replay[resume]).toHaveBeenCalledTimes(1);
      expect(replay.stop).toHaveBeenCalledTimes(1);
    },
  );

  it("leaves a replay alone between ordinary pages", async () => {
    const { Sentry, holdReplayBackFor, syncReplayWithPage } = await load();
    const replay = recordingReplay("session");
    Sentry.getReplay.mockReturnValue(replay as never);

    holdReplayBackFor("/marketplace?page=2");
    syncReplayWithPage(`${location.origin}/marketplace?page=2`);
    await settled();

    expect(replay.stop).not.toHaveBeenCalled();
    expect(replay.start).not.toHaveBeenCalled();
  });

  it("does not resume once consent is withdrawn", async () => {
    const { Sentry, holdReplayBackFor, syncReplayWithPage } = await load();
    const replay = recordingReplay("session");
    Sentry.getReplay.mockReturnValue(replay as never);

    holdReplayBackFor("/link/t-1");
    answerCookiebot({ statistics: false });
    syncReplayWithPage(`${location.origin}/library`);
    await settled();

    expect(replay.stop).toHaveBeenCalledTimes(1);
    expect(replay.start).not.toHaveBeenCalled();
  });

  it("does not start a replay that was not recording", async () => {
    const { Sentry, holdReplayBackFor, syncReplayWithPage } = await load();
    const replay = { ...recordingReplay("buffer"), getReplayId: vi.fn() };
    Sentry.getReplay.mockReturnValue(replay as never);

    holdReplayBackFor("/link/t-1");
    syncReplayWithPage(`${location.origin}/library`);
    await settled();

    expect(replay.stop).not.toHaveBeenCalled();
    expect(replay.startBuffering).not.toHaveBeenCalled();
  });

  it("restarts only once a slow stop has settled, on the page it lands on", async () => {
    const { Sentry, holdReplayBackFor, syncReplayWithPage } = await load();
    const replay = recordingReplay("session");
    let finishStop = () => {};
    replay.stop.mockReturnValue(
      new Promise<undefined>((resolve) => {
        finishStop = () => resolve(undefined);
      }),
    );
    Sentry.getReplay.mockReturnValue(replay as never);

    holdReplayBackFor("/link/t-1");
    history.replaceState({}, "", "/library");
    syncReplayWithPage(location.href);
    await settled();
    expect(replay.start).not.toHaveBeenCalled();

    // The visitor is back on a credential page by the time the stop settles.
    history.replaceState({}, "", "/link/t-2");
    finishStop();
    await settled();
    expect(replay.start).not.toHaveBeenCalled();

    history.replaceState({}, "", "/library");
    syncReplayWithPage(location.href);
    await settled();
    expect(replay.start).toHaveBeenCalledTimes(1);
  });
});
