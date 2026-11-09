import type { Page } from "@playwright/test";
import {
  currentSessionID,
  finishedChatStream,
  recordChatStreams,
  savedAssistantTexts,
  savedMessages,
  startChat,
} from "./copilot-chat";
import { expect, test } from "./fixtures";
import buildDoublerScript from "./model-scripts/build-doubler-agent.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
  streamedTexts,
} from "./scripted-model";

// The model reads the building guide, saves the agent with create_agent and
// says it's done; AutoPilot runs both tools for real.
const SCRIPT: ModelScript = buildDoublerScript;
const AGENT_NAME = "Journey Doubler";

test.describe("copilot build agent journey", () => {
  test.beforeAll(requireScriptedModel);

  test("AutoPilot builds an agent: the streamed answer is what's saved, the agent is in the Library, and the chat costs what the model billed", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("build-doubler-agent");
    const modelTexts = SCRIPT.turns.map((turn) => turn.text ?? "");
    expect(JSON.stringify(SCRIPT), "the script builds the agent").toContain(
      AGENT_NAME,
    );

    await recordChatStreams(page);
    await startChat(
      page,
      `Build me an agent that doubles a number. ${run.marker}`,
    );
    const streamed = streamedTexts(await finishedChatStream(page, "POST"));
    expect(streamed, "the stream should carry what the model said").toEqual(
      modelTexts,
    );
    const sessionID = await currentSessionID(page);

    await expect
      .poll(() => savedAssistantTexts(page, sessionID), {
        message: "the saved chat should hold exactly what was streamed",
      })
      .toEqual(streamed);
    await page.reload();
    const transcript = page.getByRole("log");
    for (const text of streamed) {
      await expect(
        transcript.getByText(text, { exact: true }),
        "the reloaded chat should show what was streamed, once",
      ).toHaveCount(1);
    }

    const saved = await savedAgent(page, sessionID);
    expect(saved.agent_name).toBe(AGENT_NAME);
    await page.goto("/library");
    await expect(
      page
        .getByTestId("library-agent-card")
        .filter({ hasText: AGENT_NAME })
        .and(page.locator(`[data-agent-id="${saved.library_agent_id}"]`)),
      "the Library should show the agent the chat saved",
    ).toBeVisible();
    expect(
      await libraryAgentsNamed(page, AGENT_NAME),
      "the chat should save the agent once",
    ).toHaveLength(1);

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, SCRIPT);
    await expectChargedWhatWasBilled(userID(journeyUser), journal);
  });
});

function userID(user: { id?: string }) {
  if (!user.id) throw new Error("the journey user should have an id");
  return user.id;
}

async function savedAgent(page: Page, sessionID: string) {
  const results = (await savedMessages(page, sessionID))
    .filter((message) => message.role === "tool" && message.content)
    .map((message) => parseJSON(message.content as string))
    .filter((content) => content?.type === "agent_builder_saved");
  expect(results, "the chat should record one saved agent").toHaveLength(1);
  return results[0] as { agent_name: string; library_agent_id: string };
}

async function libraryAgentsNamed(page: Page, name: string) {
  const response = await page.request.get("/api/proxy/api/library/agents", {
    params: { search_term: name },
  });
  expect(response.ok(), `GET library agents: ${response.status()}`).toBe(true);
  const body = (await response.json()) as { agents: { name: string }[] };
  return body.agents.filter((agent) => agent.name === name);
}

function parseJSON(text: string) {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}
