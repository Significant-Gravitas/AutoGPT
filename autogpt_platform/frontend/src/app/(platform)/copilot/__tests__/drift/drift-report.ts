import { server } from "@/mocks/mock-server";
import { cleanup, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { expect, type Mock } from "vitest";
import { resetCopilotChatRegistry } from "../../copilotChatRegistry";
import { useCopilotStreamStore } from "../../copilotStreamStore";
import { renderHost } from "../sse-helpers";
import {
  type BackendSim,
  type RecordedTurn,
  type Row,
  STREAM_URL,
  streamedTextBlocks,
} from "./backend-sim";

interface TranscriptEntry {
  role: "user" | "assistant";
  text: string;
}

/** The invariants every case is held to: `null` when one holds, what the
 *  user saw when it does not. */
interface DriftReport {
  /** Every cause here is an expected disconnect, handled silently. */
  connectionToast: string[] | null;
  /** No streamed text block is ever on screen twice. */
  paintedTwice: Record<string, number> | null;
  /** Once settled, the chat shows what a reload from the persisted rows shows. */
  differsFromReload: {
    live: TranscriptEntry[];
    reloaded: TranscriptEntry[];
  } | null;
  /** Text the user watched stream in is still there once the turn ends. */
  lostStreamedText: string[] | null;
  /** Text the stream holds intact is on screen while the turn still runs. */
  missingWhileRunning: string[] | null;
  /** A turn the backend finished cleanly is not shown as failed. */
  errorShown: string | null;
}

type Invariant = keyof DriftReport;

const ERROR_BOX_TEXT = "The assistant encountered an error";

/** Every transcript the page paints, one sample per DOM mutation. */
export function sampleTranscripts() {
  const samples: TranscriptEntry[][] = [];
  const observer = new MutationObserver(() => samples.push(readTranscript()));
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });
  return {
    samples,
    stop: () => observer.disconnect(),
  };
}

/** Which of `texts` the transcript does not show right now. */
export function notOnScreen(texts: string[]) {
  const shown = readTranscript()
    .map((entry) => entry.text)
    .join("\n");
  return texts.filter((text) => !shown.includes(text));
}

/**
 * Settle the live chat, then reload from the persisted rows and compare.
 * Unmounts the live chat, so it is the last step of a case.
 */
export async function reportDrift({
  sim,
  turns,
  painted,
  toast,
  missingWhileRunning = [],
  quietMs = 2500,
}: {
  sim: BackendSim;
  turns: RecordedTurn[];
  painted: ReturnType<typeof sampleTranscripts>;
  toast: Mock;
  /** From `notOnScreen`, taken while the turn ran. */
  missingWhileRunning?: string[];
  quietMs?: number;
}): Promise<DriftReport> {
  const live = await waitForStableTranscript(quietMs, 60_000);
  painted.stop();
  const blocks = turns.flatMap(streamedTextBlocks);
  const shownText = live.map((entry) => entry.text).join("\n");
  const paintedTwice = Object.fromEntries(
    blocks
      .map((block) => [block, maxTimesPainted(painted.samples, block)] as const)
      .filter(([, times]) => times > 1),
  );
  const lost = blocks.filter(
    (block) =>
      maxTimesPainted(painted.samples, block) > 0 && !shownText.includes(block),
  );
  const toasts = connectionToasts(toast);
  const errorShown = document.body.textContent?.includes(ERROR_BOX_TEXT);
  const reloaded = await renderReloaded(sim.finalRows());
  return {
    connectionToast: toasts.length ? toasts : null,
    paintedTwice: Object.keys(paintedTwice).length ? paintedTwice : null,
    differsFromReload:
      JSON.stringify(live) === JSON.stringify(reloaded)
        ? null
        : { live, reloaded },
    lostStreamedText: lost.length ? lost : null,
    missingWhileRunning: missingWhileRunning.length
      ? missingWhileRunning
      : null,
    errorShown: errorShown ? ERROR_BOX_TEXT : null,
  };
}

/**
 * Hold a case to every invariant except the ones `today` names as broken by
 * a live bug. A named invariant that holds again fails the case, so a fix
 * shows up as a red test that asks for its entry to be removed.
 */
export function expectDrift(
  report: DriftReport,
  {
    today = [],
    unchecked = [],
  }: { today?: Invariant[]; unchecked?: Invariant[] },
) {
  const broken = (Object.keys(report) as Invariant[]).filter(
    (name) => report[name] !== null,
  );
  const fixed = today.filter((name) => report[name] === null);
  const unexpected = broken.filter(
    (name) => !today.includes(name) && !unchecked.includes(name),
  );
  expect(
    {
      unexpected: Object.fromEntries(unexpected.map((n) => [n, report[n]])),
      fixed,
    },
    fixed.length
      ? `Now holds: ${fixed.join(", ")}. Remove it from this case's \`today\`.`
      : "An invariant broke that this case does not expect to break.",
  ).toEqual({ unexpected: {}, fixed: [] });
}

export function resetChatState() {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
}

/** Wait until the transcript has not changed for `quietMs`. */
export async function waitForStableTranscript(quietMs = 1500, timeout = 15000) {
  let last = JSON.stringify(readTranscript());
  let since = Date.now();
  await waitFor(
    () => {
      const now = JSON.stringify(readTranscript());
      if (now !== last) {
        last = now;
        since = Date.now();
      }
      expect(Date.now() - since).toBeGreaterThanOrEqual(quietMs);
    },
    { timeout, interval: 100 },
  );
  return readTranscript();
}

/** The oracle: what a fresh page load renders from the persisted rows alone. */
async function renderReloaded(rows: Row[]) {
  cleanup();
  resetChatState();
  server.use(
    http.get(STREAM_URL, () => new HttpResponse(null, { status: 204 })),
  );
  renderHost({ sessionOverride: { messages: rows } });
  await waitFor(() => expect(readTranscript().length).toBeGreaterThan(0), {
    timeout: 10000,
  });
  return waitForStableTranscript(500);
}

/**
 * One entry per rendered message, in order. The turn's "Thought for" label is
 * left out: the client counts it live and the server stamps its own figure.
 */
function readTranscript(): TranscriptEntry[] {
  return Array.from(document.querySelectorAll("[data-message-id]")).map(
    (el) => ({
      role: el.classList.contains("is-user") ? "user" : "assistant",
      text: (el.textContent ?? "")
        .replace(/Thought for (\d+m )?\d+s/g, "")
        .replace(/\s+/g, " ")
        .trim(),
    }),
  );
}

function maxTimesPainted(samples: TranscriptEntry[][], text: string) {
  return Math.max(
    0,
    ...samples.map(
      (sample) =>
        sample
          .map((entry) => entry.text)
          .join("\n")
          .split(text).length - 1,
    ),
  );
}

function connectionToasts(toast: Mock) {
  return toast.mock.calls
    .map(([options]) => options ?? {})
    .filter(({ title }) => String(title).startsWith("Connection"))
    .map(({ title, description }) => `${title}: ${description}`);
}
