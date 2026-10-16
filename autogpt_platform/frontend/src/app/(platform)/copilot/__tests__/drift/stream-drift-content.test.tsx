import { afterEach, beforeEach, describe, it, vi } from "vitest";
import { TEST_BACKEND_BASE_URL } from "../sse-helpers";
import {
  createBackendSim,
  loadRecordedTurn,
  renderAgainst,
  sendPrompt,
  turnKeepsRunning,
  waitForConnections,
} from "./backend-sim";
import {
  expectDrift,
  notOnScreen,
  reportDrift,
  resetChatState,
  sampleTranscripts,
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

const toolTurn = loadRecordedTurn("baseline-tool-turn");
const textTurn = loadRecordedTurn("dummy-text-turn");
// Frame 6 is the first block's second delta, 12 the tool's output, and
// 13-17 the whole second text block.
const SECOND_DELTA = 6;
const TOOL_OUTPUT_DONE = 13;
const SECOND_BLOCK_DONE = 18;

beforeEach(() => {
  resetChatState();
  toastMock.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  resetChatState();
});

describe("stream drift — what the stream and the rows hold", () => {
  it(
    "W5: a reload into a turn whose stream lost its head still renders the turn",
    { timeout: 90_000 },
    async () => {
      const sim = createBackendSim([{ turn: toolTurn }]);
      sim.beginRunning();
      sim.publish(TOOL_OUTPUT_DONE);
      // Simulated: the length cap dropped the stream's first entries, so the
      // replay opens on a text-delta whose text-start is gone.
      sim.trimHead(SECOND_DELTA);
      renderAgainst(sim);
      const painted = sampleTranscripts();
      await waitForConnections(sim, 1);
      // The second block reaches the stream whole, start included.
      sim.publish(SECOND_BLOCK_DONE - TOOL_OUTPUT_DONE);
      await turnKeepsRunning();
      const missingWhileRunning = notOnScreen(["It says Example Domain."]);
      sim.publish();

      expectDrift(
        await reportDrift({
          sim,
          turns: [toolTurn],
          painted,
          toast: toastMock,
          missingWhileRunning,
          quietMs: 8000,
        }),
        // The parser throws on the orphan delta and every reconnect replays
        // the same head, so nothing of the running turn renders and the
        // reconnects announce "Connection lost". Only the turn's end repairs
        // the transcript, from the rows.
        { today: ["connectionToast", "missingWhileRunning"] },
      );
    },
  );

  it(
    "approval wake: a reload into a turn no executor runs raises no connection alarm",
    { timeout: 60_000 },
    async () => {
      const [prompt] = textTurn.rows;
      // The executor dropped the turn, so its stream never gets an entry.
      const sim = createBackendSim([{ turn: { frames: [], rows: [prompt] } }]);
      sim.beginRunning();
      vi.useFakeTimers({ shouldAdvanceTime: true });
      renderAgainst(sim);
      const painted = sampleTranscripts();
      await waitForConnections(sim, 1);
      // Heartbeats keep arriving every 10 s: the connection is alive, only
      // the turn is dead, and only the server can say so.
      await vi.advanceTimersByTimeAsync(60_000);

      expectDrift(
        await reportDrift({ sim, turns: [], painted, toast: toastMock }),
        // The 6 s restore watchdog sees no content, reconnects three times and
        // ends on "Unable to reconnect. Please refresh the page."
        { today: ["connectionToast"] },
      );
    },
  );

  it(
    "W8: a reply whose final persist failed stays on screen",
    { timeout: 60_000 },
    async () => {
      const [prompt] = textTurn.rows;
      const sim = createBackendSim([
        { turn: textTurn, persistedRows: [prompt] },
      ]);
      renderAgainst(sim);
      await sendPrompt(sim, textTurn);
      const painted = sampleTranscripts();
      sim.publish();

      expectDrift(
        await reportDrift({
          sim,
          turns: [textTurn],
          painted,
          toast: toastMock,
        }),
        // The end-of-turn hydrate replaces the list with rows that lack the
        // reply, so it vanishes in front of the user. The rows are the fault
        // here, so a reload is no oracle for this case.
        { today: ["lostStreamedText"], unchecked: ["differsFromReload"] },
      );
    },
  );
});
