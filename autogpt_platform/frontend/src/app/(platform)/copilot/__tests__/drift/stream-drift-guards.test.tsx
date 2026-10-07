import { screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STREAM_PATHS, TEST_BACKEND_BASE_URL } from "../sse-helpers";
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
const streamPath = vi.hoisted(() => ({ runtime: false }));
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "copilot-stream-runtime" ? streamPath.runtime : false,
  };
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

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

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

    // The AI SDK path swaps every id for the rows' `-seq-N` ones at the end
    // of the turn, which remounts the transcript; the runtime adopts the rows
    // into the messages it already drew.
    const runtimeOnly = path === "stream runtime" ? it : it.fails;
    runtimeOnly(
      "the end-of-turn reconcile keeps every message's element",
      { timeout: 60_000 },
      async () => {
        const turn = loadRecordedTurn("baseline-tool-turn");
        const sim = createBackendSim([{ turn }]);
        renderAgainst(sim);
        await sendPrompt(sim, turn);
        sim.publish(turn.frames.length - 1);
        await waitForStableTranscript(1000);
        const before = messageElements();

        sim.publish();
        // The reconcile adopts the prompt's persisted row, timestamp included.
        const stamp = new Date(String(turn.rows[0].created_at)).toLocaleString(
          undefined,
          { dateStyle: "medium", timeStyle: "short" },
        );
        await waitFor(() => expect(screen.getByText(stamp)).toBeDefined(), {
          timeout: 10_000,
        });
        await waitForStableTranscript(1000);
        const after = messageElements();
        expect(after).toHaveLength(2);
        expect(after.filter((element) => !before.includes(element))).toEqual(
          [],
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
});

function messageElements() {
  return Array.from(document.querySelectorAll("[data-message-id]"));
}
