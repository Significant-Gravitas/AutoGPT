import { randomBytes } from "crypto";
import { expect, test } from "@playwright/test";
import { withJourneyDatabase } from "./journey-database";

/**
 * Journeys that chat with AutoPilot run against the scripted model
 * (autogpt_platform/scripted_model): the real copilot pipeline, with model
 * turns played from model-scripts/<name>.json. The stack must be started with
 * docker-compose.journeys.yml, which CI's journeys leg does.
 */
const SCRIPTED_MODEL_URL =
  process.env.PLAYWRIGHT_SCRIPTED_MODEL_URL ?? "http://localhost:4010";

export interface ModelScript {
  title?: string;
  turns: {
    text?: string;
    tool_calls?: { name: string; input: Record<string, unknown> }[];
    expect_user_text?: string;
  }[];
}

export interface JournalEntry {
  kind: "turn" | "side_call" | "error" | "unexpected";
  model?: string;
  turn?: number | null;
  error?: string;
  cost_usd?: number | null;
  checks?: { what: string; ok: boolean; detail: string }[];
  completed?: boolean | null;
}

/** A chat message that makes the model follow `script`, and its run's id. */
export function scriptedRun(script: string) {
  const nonce = randomBytes(6).toString("hex");
  return { nonce, marker: `[[script:${script}#${nonce}]]` };
}

/**
 * Skips the journey against a deployed app, which talks to real models.
 * On a local stack, fails unless the scripted model is up.
 */
export async function requireScriptedModel() {
  const { hostname } = new URL(
    process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
  );
  test.skip(
    !["localhost", "127.0.0.1", "[::1]"].includes(hostname),
    "scripted model journeys need a local stack with docker-compose.journeys.yml",
  );
  const healthy = await fetch(`${SCRIPTED_MODEL_URL}/healthz`)
    .then((response) => response.ok)
    .catch(() => false);
  if (!healthy) {
    throw new Error(
      `the scripted model isn't answering at ${SCRIPTED_MODEL_URL}. Start ` +
        "the stack with docker-compose.journeys.yml (see scripted_model/README.md).",
    );
  }
}

/** Every model request a run made, oldest first. */
export async function readJournal(nonce: string) {
  const response = await fetch(
    `${SCRIPTED_MODEL_URL}/__scripted__/journal?nonce=${nonce}`,
  );
  expect(response.ok, `GET journal: ${response.status}`).toBe(true);
  const body = (await response.json()) as { entries: JournalEntry[] };
  return body.entries;
}

/**
 * Checks the platform asked for every scripted turn exactly once and each
 * reply reached it whole: no refusal, no retried or repeated turn, and every
 * tool result the script expected.
 */
export function expectEachTurnOnce(
  journal: JournalEntry[],
  script: ModelScript,
) {
  const turns = journal.filter((entry) => entry.kind === "turn");
  const problems = journal.filter(
    (entry) => entry.kind === "error" || entry.kind === "unexpected",
  );
  expect(problems, "the model should answer every request").toEqual([]);
  expect(
    turns.map((entry) => entry.turn),
    "the platform should ask for each scripted turn exactly once",
  ).toEqual(script.turns.map((_, index) => index + 1));
  expect(
    turns.flatMap((entry) => entry.checks ?? []).filter((check) => !check.ok),
    "every tool result and user message should be what the script expected",
  ).toEqual([]);
  expect(
    turns.filter((entry) => entry.completed !== true),
    "every reply should reach the platform in full",
  ).toEqual([]);
}

/**
 * Checks the copilot's cost ledger charged the user exactly what the model
 * billed the runs (`nonces`) for their turns and side calls (the chat's
 * title): every request once, none missed, none twice. Each ledger row may be
 * off by rounding, as the backend's billing audit allows. The journal and the
 * ledger are re-read until they agree, and only once every run has asked for
 * its title, which happens in the background.
 *
 * The ledger is the user's whole copilot ledger, so it needs a brand-new user;
 * an existing account (PLAYWRIGHT_JOURNEY_EMAIL) carries earlier chats'
 * charges, and the check is skipped there.
 */
export async function expectChargedWhatWasBilled(
  userID: string,
  nonces: string[],
) {
  test.skip(
    Boolean(process.env.PLAYWRIGHT_JOURNEY_EMAIL),
    "the charge check reads the user's whole cost ledger, so it needs a brand-new user",
  );
  await expect(async () => {
    const journals = await Promise.all(nonces.map(readJournal));
    // Compared before a title went out, the totals would agree without it.
    expect(
      nonces.filter(
        (_, run) => !journals[run].some((entry) => entry.kind === "side_call"),
      ),
      "every chat should have asked for its title",
    ).toEqual([]);
    const billed = journals
      .flat()
      .filter((entry) => entry.kind === "turn" || entry.kind === "side_call")
      .reduce(
        (sum, entry) => sum + Math.round((entry.cost_usd ?? 0) * 1_000_000),
        0,
      );
    expect(billed, "the model should bill the run").toBeGreaterThan(0);
    const rows = await copilotCostRows(userID);
    const charged = rows.reduce((sum, row) => sum + row.costMicrodollars, 0);
    expect(
      Math.abs(charged - billed),
      `charged ${charged} microdollars for requests billed ${billed}: ` +
        JSON.stringify(rows),
    ).toBeLessThanOrEqual(2 * Math.max(rows.length, 1));
  }, "the chat should be charged exactly what the model billed").toPass({
    timeout: 20_000,
  });
}

async function copilotCostRows(userID: string) {
  return withJourneyDatabase("reading the copilot cost ledger", async (db) => {
    const result = await db.query<{
      blockName: string;
      costMicrodollars: string | null;
      chatSessionId: string | null;
    }>(
      `SELECT "blockName", "costMicrodollars"::text AS "costMicrodollars", "chatSessionId"
       FROM platform."PlatformCostLog"
       WHERE "userId" = $1 AND "metadata"->>'source' = 'copilot'
       ORDER BY "createdAt"`,
      [userID],
    );
    return result.rows.map((row) => ({
      ...row,
      costMicrodollars: Number(row.costMicrodollars ?? 0),
    }));
  });
}

/**
 * The text a chat stream carried, one entry per text block, as the browser
 * received it.
 */
export function streamedTexts(body: string) {
  const texts = new Map<string, string>();
  for (const line of body.split("\n")) {
    if (!line.startsWith("data:")) continue;
    const event = parseEvent(line.slice("data:".length).trim());
    if (event?.type === "text-delta" && event.id) {
      texts.set(event.id, (texts.get(event.id) ?? "") + (event.delta ?? ""));
    }
  }
  return [...texts.values()];
}

function parseEvent(data: string) {
  try {
    return JSON.parse(data) as { type?: string; id?: string; delta?: string };
  } catch {
    return null;
  }
}
