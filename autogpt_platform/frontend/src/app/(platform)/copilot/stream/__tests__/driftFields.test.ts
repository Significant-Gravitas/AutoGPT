import { beforeAll, describe, expect, it } from "vitest";
import * as Sentry from "@sentry/nextjs";
import { diffReportLines, diffRows } from "../turnShadow";
import type { LogRow } from "../turnLog";

// The real Sentry SDK, with a beforeSend that captures the event after the
// SDK's `normalizeDepth` (default 3) ran, and a stubbed transport so nothing
// leaves the process. This is the BUILDER-95J failure mode: with drift rows
// reported as `[{ index, fields }]`, every `fields` array died at depth 3 and
// arrived on Sentry as `"[Array]"`, so the events named no field.
const sentryEvents: Array<{ extra?: Record<string, unknown> }> = [];

beforeAll(() => {
  Sentry.init({
    dsn: "https://aabbccddeeff00112233445566778899@o123456.ingest.sentry.io/123456",
    transport: () =>
      ({
        send: async () => ({ statusCode: 200 }),
        flush: async () => true,
      }) as unknown as ReturnType<
        NonNullable<Parameters<typeof Sentry.init>[0]["transport"]>
      >,
    beforeSend: (event) => {
      sentryEvents.push(event as { extra?: Record<string, unknown> });
      return null;
    },
  });
});

function row(partial: Partial<LogRow>): LogRow {
  return {
    key: "row",
    role: "assistant",
    content: "",
    toolCalls: [],
    toolCallId: null,
    sequence: null,
    metadata: null,
    ...partial,
  };
}

describe("copilot stream drift reports", () => {
  it("survives Sentry normalization with the field names intact", async () => {
    const live: LogRow[] = [
      row({ key: "k0", role: "user", content: "What is 2+2?" }),
      row({ key: "k1", content: "It says 4." }),
      row({ key: "k2", content: "It says 5." }),
      row({
        key: "k3",
        toolCallId: "call-1",
        role: "tool",
        content: '{"n": 1}',
        toolCalls: [
          { id: "call-1", name: "calc", input: 2, displayName: null },
        ],
      }),
      row({ key: "k4", content: "Done." }),
    ];
    const persisted: LogRow[] = [
      live[0],
      live[1],
      row({ key: "k2", content: "It says Other." }),
      row({
        key: "k3",
        toolCallId: "call-1",
        role: "tool",
        content: '{"n": 2}',
        toolCalls: [
          { id: "call-1", name: "calc", input: 2, displayName: null },
        ],
      }),
      live[4],
    ];

    const diffs = diffRows(live, persisted);
    expect(diffs).toEqual([
      { index: 2, fields: ["content"] },
      { index: 3, fields: ["output"] },
    ]);

    // What the two drift reporters send: one flat string per differing row.
    Sentry.captureMessage("copilot stream drift: finish", {
      level: "warning",
      tags: { stream_drift: "finish" },
      extra: {
        turnId: "drift-turn",
        cursor: "0",
        diffs: diffReportLines(diffs),
      },
    });
    await Sentry.flush(5_000);

    expect(sentryEvents.length).toBeGreaterThan(0);
    const event = sentryEvents[sentryEvents.length - 1];
    expect(event.extra).toMatchObject({
      turnId: "drift-turn",
      diffs: ["2: content", "3: output"],
    });
  });

  it("keeps the old deep shape failing on Sentry, so a revert cannot sneak back in", async () => {
    Sentry.captureMessage("copilot stream drift: finish", {
      level: "warning",
      tags: { stream_drift: "finish" },
      extra: { diffs: [{ index: 0, fields: ["content"] }] },
    });
    await Sentry.flush(5_000);

    const event = sentryEvents[sentryEvents.length - 1];
    const diffs = event.extra?.diffs as Array<{ fields: string }>;
    expect(diffs[0].fields).toBe("[Array]");
  });
});
