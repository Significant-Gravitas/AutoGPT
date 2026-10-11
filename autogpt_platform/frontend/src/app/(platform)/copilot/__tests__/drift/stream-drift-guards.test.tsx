import { afterEach, beforeEach, describe, it, vi } from "vitest";
import { TEST_BACKEND_BASE_URL } from "../sse-helpers";
import {
  createBackendSim,
  loadRecordedTurn,
  renderAgainst,
  sendPrompt,
  turnKeepsRunning,
  streamedTextBlocks,
  waitForConnections,
} from "./backend-sim";
import {
  expectDrift,
  notOnScreen,
  reportDrift,
  resetChatState,
  sampleTranscripts,
  waitForStableTranscript,
} from "./drift-report";

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
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});
const { toastMock } = vi.hoisted(() => ({ toastMock: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

beforeEach(() => {
  resetChatState();
  toastMock.mockClear();
});

afterEach(() => resetChatState());

describe("stream drift — guards", () => {
  it.each([
    "dummy-text-turn",
    "baseline-tool-turn",
    // A tool result that lands after the next text: the backend ends the
    // text block before the step closes (#14762), or the parser dies.
    "sdk-late-tool-result",
  ])(
    "a sent turn renders what a reload renders (%s)",
    { timeout: 60_000 },
    async (name) => {
      const turn = loadRecordedTurn(name);
      const sim = createBackendSim([{ turn }]);
      renderAgainst(sim);
      await sendPrompt(sim, turn);
      const painted = sampleTranscripts();
      sim.publish(turn.frames.length - 1);
      await waitForStableTranscript(1000);
      const missingWhileRunning = notOnScreen(streamedTextBlocks(turn));
      sim.publish();

      expectDrift(
        await reportDrift({
          sim,
          turns: [turn],
          painted,
          toast: toastMock,
          missingWhileRunning,
        }),
        {},
      );
    },
  );

  it(
    "a reload into a running turn resumes it once",
    { timeout: 60_000 },
    async () => {
      const turn = loadRecordedTurn("baseline-tool-turn");
      const sim = createBackendSim([{ turn }]);
      sim.beginRunning();
      sim.publish(9);
      renderAgainst(sim);
      const painted = sampleTranscripts();
      await waitForConnections(sim, 1);
      await turnKeepsRunning();
      sim.publish(turn.frames.length - 1 - 9);
      await waitForStableTranscript(1000);
      const missingWhileRunning = notOnScreen(streamedTextBlocks(turn));
      sim.publish();

      expectDrift(
        await reportDrift({
          sim,
          turns: [turn],
          painted,
          toast: toastMock,
          missingWhileRunning,
        }),
        {},
      );
    },
  );
});
