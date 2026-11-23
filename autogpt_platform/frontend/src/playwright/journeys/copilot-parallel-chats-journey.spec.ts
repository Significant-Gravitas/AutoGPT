import type { APIRequestContext } from "@playwright/test";
import { expect, test } from "./fixtures";
import quickAnswerScript from "./model-scripts/quick-answer.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type JournalEntry,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
  streamedTexts,
} from "./scripted-model";

// The copilot executor runs chats on several workers, each with its own event
// loop. When those loops shared one model client, a request could go out on a
// connection another loop had opened, fail after it was sent, and be retried:
// the model answered and billed it twice while the platform recorded it once.
// Chats started together, in two waves so the second finds the idle
// connections the first left behind, must each ask the model for their title
// once and for each turn once, and be charged exactly what the model billed.
//
// The chats go through the chat API the composer uses, not through several
// browsers: the invariant is the backend's, and many pages loading /home at
// once trip unrelated frontend errors.
const SCRIPT: ModelScript = quickAnswerScript;
const ANSWER = SCRIPT.turns[0].text ?? "";
const CHATS_PER_WAVE = 4;
const WAVES = 2;

test.describe("copilot parallel chats journey", () => {
  test.beforeAll(requireScriptedModel);

  test("chats started together each ask the model once per request", async ({
    page,
    journeyUser,
  }) => {
    const chats: { nonce: string; sessionID: string }[] = [];
    for (let wave = 0; wave < WAVES; wave++) {
      chats.push(
        ...(await Promise.all(
          Array.from({ length: CHATS_PER_WAVE }, () =>
            chatToTitle(page.request),
          ),
        )),
      );
    }

    const journals = await Promise.all(
      chats.map((chat) => readJournal(chat.nonce)),
    );
    expect(
      journals.map((journal) => titleRequests(journal).length),
      "each chat should ask the model for its title once",
    ).toEqual(chats.map(() => 1));
    for (const journal of journals) expectEachTurnOnce(journal, SCRIPT);
    if (!journeyUser.id) throw new Error("the journey user should have an id");
    await expectChargedWhatWasBilled(
      journeyUser.id,
      chats.map((chat) => chat.nonce),
    );
  });
});

/** Starts a chat and waits for its answer and its title from the model. */
async function chatToTitle(api: APIRequestContext) {
  const run = scriptedRun("quick-answer");
  const created = await api.post("/api/proxy/api/chat/sessions", { data: {} });
  expect(created.ok(), `POST chat session: ${created.status()}`).toBe(true);
  const { id: sessionID } = (await created.json()) as { id: string };
  const turn = await api.post(
    `/api/proxy/api/chat/sessions/${sessionID}/stream`,
    { data: { message: `Give me a short answer. ${run.marker}` } },
  );
  expect(turn.ok(), `POST chat stream: ${turn.status()}`).toBe(true);
  expect(
    streamedTexts(await turn.text()),
    "the answer should stream back",
  ).toContain(ANSWER);
  // The title is saved after the title request returns, retries included,
  // so the journal is complete for this chat once it shows.
  await expect
    .poll(() => savedTitle(api, sessionID), {
      message: "the chat should be titled by the model",
      timeout: 30_000,
    })
    .toBe(SCRIPT.title);
  return { nonce: run.nonce, sessionID };
}

/** The chat's requests without tools: its title is the only one. */
function titleRequests(journal: JournalEntry[]) {
  return journal.filter((entry) => entry.kind === "side_call");
}

async function savedTitle(api: APIRequestContext, sessionID: string) {
  const response = await api.get("/api/proxy/api/chat/sessions?limit=100");
  expect(response.ok(), `GET chat sessions: ${response.status()}`).toBe(true);
  const body = (await response.json()) as {
    sessions: { id: string; title: string | null }[];
  };
  return body.sessions.find((session) => session.id === sessionID)?.title;
}
