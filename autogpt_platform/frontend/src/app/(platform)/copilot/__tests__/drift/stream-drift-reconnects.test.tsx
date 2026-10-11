import { screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STREAM_PATHS, TEST_BACKEND_BASE_URL } from "../sse-helpers";
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

const toolTurn = loadRecordedTurn("baseline-tool-turn");
const textTurn = loadRecordedTurn("dummy-text-turn");
// Frames 0-8 end the first text block; 10-11 open the tool call.
const FIRST_TEXT_BLOCK_DONE = 9;
const TOOL_CALL_OPEN = 12;

beforeEach(() => {
  resetChatState();
  toastMock.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  setTabVisibility("visible");
  resetChatState();
});

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  describe("stream drift — a second connection onto a live turn", () => {
    // Both paths open one connection. The AI SDK path paints the first block
    // twice: the replay after a cut re-draws it (as in W4).
    const runtimeOnly = path === "stream runtime" ? it : it.fails;
    runtimeOnly(
      "W1: two resumes inside the window open one connection and draw one bubble",
      { timeout: 60_000 },
      async () => {
        const sim = createBackendSim([{ turn: toolTurn }]);
        renderAgainst(sim);
        await sendPrompt(sim, toolTurn);
        const painted = sampleTranscripts();
        sim.publish(FIRST_TEXT_BLOCK_DONE);
        await screen.findByText("Let me fetch that page.", undefined, {
          timeout: 5000,
        });

        // The stream drops while the tab is hidden. Coming back after 30 s
        // resumes it; the network event that follows is a second trigger
        // inside the liveness window of a resume already on the wire.
        vi.useFakeTimers({ toFake: ["Date"], shouldAdvanceTime: true });
        setTabVisibility("hidden");
        sim.cutOpenConnections();
        vi.setSystemTime(Date.now() + 31_000);
        setTabVisibility("visible");
        vi.useRealTimers();
        await waitForConnections(sim, 2);
        window.dispatchEvent(new Event("online"));
        await turnKeepsRunning();
        sim.publish();

        const report = await reportDrift({
          sim,
          turns: [toolTurn],
          painted,
          toast: toastMock,
        });
        expect(sim.connections).toHaveLength(2);
        expectDrift(report, {});
      },
    );

    it(
      "W1: a tab shown again after 30 s resumes over the stream it never lost",
      { timeout: 60_000 },
      async () => {
        const sim = createBackendSim([{ turn: toolTurn }]);
        renderAgainst(sim);
        await sendPrompt(sim, toolTurn);
        const painted = sampleTranscripts();
        sim.publish(FIRST_TEXT_BLOCK_DONE);
        await screen.findByText("Let me fetch that page.", undefined, {
          timeout: 5000,
        });

        // Desktop browsers keep a hidden tab's fetch open, so the POST stream is
        // still healthy when the wake re-sync fires. The AI SDK path keeps it;
        // the runtime resumes over it in its single slot.
        vi.useFakeTimers({ toFake: ["Date"], shouldAdvanceTime: true });
        setTabVisibility("hidden");
        vi.setSystemTime(Date.now() + 31_000);
        setTabVisibility("visible");
        vi.useRealTimers();
        if (path === "stream runtime") await waitForConnections(sim, 2);
        else await turnKeepsRunning();
        sim.publish();

        expectDrift(
          await reportDrift({
            sim,
            turns: [toolTurn],
            painted,
            toast: toastMock,
          }),
          {},
        );
      },
    );

    it(
      "W7: a tool call silent for 70 s keeps its stream, heartbeats included",
      { timeout: 60_000 },
      async () => {
        const sim = createBackendSim([{ turn: toolTurn }]);
        renderAgainst(sim);
        await sendPrompt(sim, toolTurn);
        const painted = sampleTranscripts();
        sim.publish(FIRST_TEXT_BLOCK_DONE);
        await screen.findByText("Let me fetch that page.", undefined, {
          timeout: 5000,
        });

        vi.useFakeTimers({ shouldAdvanceTime: true });
        sim.publish(TOOL_CALL_OPEN - FIRST_TEXT_BLOCK_DONE);
        // The route writes a heartbeat comment every 10 s; the parser drops them.
        await vi.advanceTimersByTimeAsync(70_000);
        sim.publish();

        expectDrift(
          await reportDrift({
            sim,
            turns: [toolTurn],
            painted,
            toast: toastMock,
          }),
          {},
        );
      },
    );
  });

  describe("stream drift — a stream that ends while its turn runs on", () => {
    it(
      "W4: the load balancer's 30-minute cut is resumed silently",
      { timeout: 60_000 },
      async () => {
        const sim = createBackendSim([{ turn: toolTurn }]);
        renderAgainst(sim);
        await sendPrompt(sim, toolTurn);
        const painted = sampleTranscripts();
        sim.publish(FIRST_TEXT_BLOCK_DONE);
        await screen.findByText("Let me fetch that page.", undefined, {
          timeout: 5000,
        });

        // Simulated: the cut is a clean close with no finish, whatever the age.
        sim.cutOpenConnections();
        await waitForConnections(sim, 2);
        await turnKeepsRunning();
        sim.publish();

        expectDrift(
          await reportDrift({
            sim,
            turns: [toolTurn],
            painted,
            toast: toastMock,
          }),
          { today: path === "stream runtime" ? [] : ["paintedTwice"] },
        );
      },
    );

    it(
      "chained turn: the turn a finished turn wakes is attached to quietly",
      { timeout: 60_000 },
      async () => {
        const continuation = {
          ...textTurn,
          rows: textTurn.rows.filter((row) => row.role !== "user"),
        };
        const sim = createBackendSim([
          { turn: toolTurn },
          { turn: continuation, chained: true },
        ]);
        renderAgainst(sim);
        await sendPrompt(sim, toolTurn);
        const painted = sampleTranscripts();
        // The first turn's end starts the continuation before its finish lands.
        sim.publish();
        await waitForConnections(sim, 2);
        await turnKeepsRunning();
        sim.publish();

        expectDrift(
          await reportDrift({
            sim,
            turns: [toolTurn, continuation],
            painted,
            toast: toastMock,
          }),
          {},
        );
      },
    );
  });
});

function setTabVisibility(state: "hidden" | "visible") {
  Object.defineProperty(document, "visibilityState", {
    value: state,
    configurable: true,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}
