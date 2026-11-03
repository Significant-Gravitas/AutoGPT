import * as Sentry from "@sentry/nextjs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getShadowLog, resetShadows } from "../../stream/turnShadow";
import { TEST_BACKEND_BASE_URL, TEST_SESSION_ID } from "../sse-helpers";
import {
  createBackendSim,
  loadRecordedTurn,
  renderAgainst,
  sendPrompt,
} from "./backend-sim";
import { resetChatState, waitForStableTranscript } from "./drift-report";

vi.mock("@sentry/nextjs", () => ({
  captureMessage: vi.fn(),
  captureException: vi.fn(),
  addBreadcrumb: vi.fn(),
  getTraceData: () => ({}),
}));
vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});
vi.mock("../../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../../helpers")>();
  return { ...actual, getCopilotAuthHeaders: async () => ({}) };
});
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));
const { shadowFlag } = vi.hoisted(() => ({ shadowFlag: { rate: 1 } }));
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.COPILOT_STREAM_SHADOW ? shadowFlag.rate : false,
  };
});

beforeEach(() => {
  resetChatState();
  resetShadows();
  shadowFlag.rate = 1;
  vi.mocked(Sentry.captureMessage).mockClear();
});

afterEach(() => resetChatState());

function driftKinds() {
  return vi
    .mocked(Sentry.captureMessage)
    .mock.calls.map(
      ([, hint]) =>
        (hint as { tags: { stream_drift: string } }).tags.stream_drift,
    );
}

// The shadow folds the stream the AI SDK renders, off the same response.
describe("the stream shadow inside the chat", () => {
  it.each(["dummy-text-turn", "baseline-tool-turn", "sdk-late-tool-result"])(
    "folds a sent turn to the persisted rows and reports nothing (%s)",
    { timeout: 60_000 },
    async (name) => {
      const turn = loadRecordedTurn(name);
      const sim = createBackendSim([{ turn }]);
      renderAgainst(sim);
      await sendPrompt(sim, turn);
      sim.publish();
      await waitForStableTranscript(1500);

      const log = getShadowLog(TEST_SESSION_ID);
      expect(log?.status).toBe("finished");
      expect(log?.rows.map((r) => r.content)).toEqual(
        turn.rows.slice(1).map((r) => r.content ?? ""),
      );
      expect(driftKinds()).toEqual([]);
    },
  );

  it(
    "leaves the stream alone while copilot-stream-shadow is 0",
    { timeout: 60_000 },
    async () => {
      shadowFlag.rate = 0;
      const turn = loadRecordedTurn("dummy-text-turn");
      const sim = createBackendSim([{ turn }]);
      renderAgainst(sim);
      await sendPrompt(sim, turn);
      sim.publish();
      await waitForStableTranscript(1500);

      expect(getShadowLog(TEST_SESSION_ID)).toBeNull();
    },
  );

  it(
    "W8: reports a reply whose final persist failed as drift at finish",
    { timeout: 60_000 },
    async () => {
      const turn = loadRecordedTurn("dummy-text-turn");
      const sim = createBackendSim([{ turn, persistedRows: [turn.rows[0]] }]);
      renderAgainst(sim);
      await sendPrompt(sim, turn);
      sim.publish();

      await vi.waitFor(() => expect(driftKinds()).toEqual(["finish"]), {
        timeout: 15_000,
      });
    },
  );
});
