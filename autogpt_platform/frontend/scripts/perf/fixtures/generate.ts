/**
 * Writes the synthetic copilot fixtures: a 200-row chat session, a
 * 20-line streamed reply and a reply that is mostly one long code block. Deterministic, so re-running produces the same
 * files. To capture real wire data from a running stack instead, use
 * scripts/perf/record.ts.
 *
 *   npx tsx scripts/perf/fixtures/generate.ts
 */
import fs from "fs";
import path from "path";
import { FIXTURES_DIR } from "../lib/paths";

const SESSION_ROWS = 200;
const START_TIME = Date.parse("2026-01-05T09:00:00.000Z");
const WORDS_PER_DELTA = 1;

interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface SessionRow {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  tool_calls: ToolCall[] | null;
  tool_call_id: string | null;
  sequence: number;
  duration_ms: number | null;
  created_at: string;
  metadata: Record<string, never>;
}

const TOPICS = [
  "weekly sales report",
  "customer onboarding emails",
  "competitor price tracking",
  "support ticket triage",
  "blog post outline",
  "invoice reconciliation",
  "meeting notes summary",
  "social media calendar",
];

function plainAnswer(turn: number, topic: string) {
  return [
    `Here is what I found for the ${topic}. The numbers from last week are steady, with a small rise on Thursday after the newsletter went out.`,
    "",
    `I would keep the current plan and check again on Monday. If the trend holds, we can move turn ${turn} into the regular schedule.`,
  ].join("\n");
}

function codeAnswer(turn: number, topic: string) {
  return [
    `This script prepares the ${topic} data:`,
    "",
    "```ts",
    `export function summarise(rows: { day: string; total: number }[]) {`,
    `  const total = rows.reduce((sum, row) => sum + row.total, 0);`,
    `  const best = rows.reduce((a, b) => (b.total > a.total ? b : a));`,
    `  return { total, best: best.day, turn: ${turn} };`,
    `}`,
    "```",
    "",
    "Run it once per day and paste the result here.",
  ].join("\n");
}

function tableAnswer(turn: number, topic: string) {
  return [
    `Breakdown for the ${topic}:`,
    "",
    "| Day | Items | Change |",
    "| --- | ---: | ---: |",
    `| Monday | ${120 + turn} | +4% |`,
    `| Tuesday | ${98 + turn} | -2% |`,
    `| Wednesday | ${143 + turn} | +11% |`,
    `| Thursday | ${160 + turn} | +9% |`,
    "",
    "Wednesday and Thursday carry most of the week.",
  ].join("\n");
}

function listAnswer(turn: number, topic: string) {
  return [
    `Next steps for the ${topic}:`,
    "",
    "1. Pull the latest export from the shared drive.",
    "2. Remove duplicate rows and empty lines.",
    `3. Compare against the numbers from turn ${Math.max(1, turn - 1)}.`,
    "4. Send the short summary to the team channel.",
    "",
    "- **Owner:** you",
    "- **Due:** Friday",
  ].join("\n");
}

const ANSWERS = [plainAnswer, codeAnswer, tableAnswer, listAnswer];

function rowFactory() {
  let sequence = 0;
  return function row(
    role: SessionRow["role"],
    content: string,
    extra: Partial<SessionRow> = {},
  ): SessionRow {
    const current = sequence++;
    return {
      id: `perf-msg-${String(current).padStart(4, "0")}`,
      role,
      content,
      tool_calls: null,
      tool_call_id: null,
      sequence: current,
      duration_ms: role === "assistant" ? 1800 + (current % 7) * 150 : null,
      created_at: new Date(START_TIME + current * 45_000).toISOString(),
      metadata: {},
      ...extra,
    };
  };
}

function toolTurn(
  row: ReturnType<typeof rowFactory>,
  turn: number,
  topic: string,
) {
  const callID = `call_perf_${turn}`;
  return [
    row("assistant", "", {
      tool_calls: [
        {
          id: callID,
          type: "function",
          function: {
            name: "perf_lookup",
            arguments: JSON.stringify({ query: topic, page: turn }),
          },
        },
      ],
    }),
    row("tool", JSON.stringify({ query: topic, results: 3, turn }), {
      tool_call_id: callID,
    }),
    row("assistant", plainAnswer(turn, topic)),
  ];
}

export function buildSessionRows(count = SESSION_ROWS): SessionRow[] {
  const row = rowFactory();
  const rows: SessionRow[] = [];
  for (let turn = 1; rows.length < count; turn++) {
    const topic = TOPICS[turn % TOPICS.length];
    rows.push(row("user", `Turn ${turn}: can you help with the ${topic}?`));
    if (turn % 5 === 0) {
      rows.push(...toolTurn(row, turn, topic));
    } else {
      rows.push(row("assistant", ANSWERS[turn % ANSWERS.length](turn, topic)));
    }
  }
  return rows.slice(0, count);
}

export const REPLY_20_LINES = [
  "Sure. Here is a quick plan for the launch week.",
  "",
  "## Goals",
  "",
  "- Ship the new onboarding flow to every new account.",
  "- Keep the support queue under one day of waiting time.",
  "- Collect feedback from the first fifty teams.",
  "",
  "## Script to check sign-ups",
  "",
  "```ts",
  "export function countSignups(events: { type: string }[]) {",
  '  return events.filter((event) => event.type === "signup").length;',
  "}",
  "```",
  "",
  "1. Run the script every morning and post the number.",
  "2. Flag any day that drops more than ten percent.",
  "3. Share a short summary with the team on Friday.",
  "Let me know if you want me to set up the daily check for you.",
].join("\n");

const LONG_CODE_SECTIONS = [
  "signups",
  "refunds",
  "renewals",
  "upgrades",
  "trials",
  "invites",
  "exports",
  "payouts",
  "reviews",
  "tickets",
];

function longCodeSection(name: string, index: number) {
  const title = name[0].toUpperCase() + name.slice(1);
  return [
    "/**",
    ` * Counts the ${name} events and reports the busiest day.`,
    " */",
    `export function count${title}(events: { type: string; day: string }[]) {`,
    `  const matching = events.filter((event) => event.type === "${name}");`,
    "  const byDay = new Map<string, number>();",
    "  for (const event of matching) {",
    "    byDay.set(event.day, (byDay.get(event.day) ?? 0) + 1);",
    "  }",
    `  const label = \`${name}: \${matching.length} events\`; // section ${index}`,
    "  return { total: matching.length, byDay, label };",
    "}",
    "",
  ];
}

/** A reply that is mostly one long code block, so every flush lands in it. */
export const REPLY_LONG_CODE = [
  "Here is the full report script.",
  "",
  "```ts",
  'import { readFileSync } from "fs";',
  "",
  ...LONG_CODE_SECTIONS.flatMap(longCodeSection),
  "```",
  "",
  "Run it once a day.",
].join("\n");

export function buildReplyStream(text: string, messageID: string) {
  const words = text.match(/\S+\s*|\s+/g) ?? [];
  const deltas: string[] = [];
  for (let i = 0; i < words.length; i += WORDS_PER_DELTA) {
    deltas.push(words.slice(i, i + WORDS_PER_DELTA).join(""));
  }
  const events: unknown[] = [
    { type: "start", messageId: messageID },
    { type: "start-step" },
    { type: "text-start", id: "perf-text-1" },
    ...deltas.map((delta) => ({
      type: "text-delta",
      id: "perf-text-1",
      delta,
    })),
    { type: "text-end", id: "perf-text-1" },
    { type: "finish-step" },
    { type: "finish" },
  ];
  return [
    ...events.map((event) => `data: ${JSON.stringify(event)}`),
    "data: [DONE]",
  ]
    .map((frame) => `${frame}\n\n`)
    .join("");
}

function main() {
  fs.mkdirSync(FIXTURES_DIR, { recursive: true });
  const sessionFile = path.join(FIXTURES_DIR, "session-200.json");
  fs.writeFileSync(
    sessionFile,
    `${JSON.stringify({ messages: buildSessionRows() }, null, 2)}\n`,
  );
  const replyFile = path.join(FIXTURES_DIR, "reply-20-lines.sse");
  fs.writeFileSync(
    replyFile,
    buildReplyStream(REPLY_20_LINES, "perf-reply-20-lines"),
  );
  const longCodeFile = path.join(FIXTURES_DIR, "reply-long-code.sse");
  fs.writeFileSync(
    longCodeFile,
    buildReplyStream(REPLY_LONG_CODE, "perf-reply-long-code"),
  );
  console.log(
    `wrote ${sessionFile}\nwrote ${replyFile}\nwrote ${longCodeFile}`,
  );
}

if (require.main === module) main();
