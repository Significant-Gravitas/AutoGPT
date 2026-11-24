import type { Page } from "@playwright/test";
import {
  currentSessionID,
  expectChatSettled,
  isChatStream,
  startChat,
} from "./copilot-chat";
import { expectOnceEachInOrder, savedConversation } from "./copilot-stop";
import { expect, test } from "./fixtures";
import twoTabsScript from "./model-scripts/two-tabs.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// The same chat open in two tabs, as on a phone and a desktop (#17355). The
// server tells every tab of the user, over its live connection, when a turn
// in the chat ends. The first tab hears it. The journey keeps it from the
// second, as from a tab whose connection missed it, so the second sends from
// a view that lacks the first tab's last turn.
const SCRIPT: ModelScript = twoTabsScript;
const [FIRST_ANSWER, SECOND_ANSWER, THIRD_ANSWER] = SCRIPT.turns.map(
  (turn) => turn.text ?? "",
);
const FIRST = "Lima, from the first tab.";
const SECOND = SCRIPT.turns[1].expect_user_text ?? "";
const THIRD = SCRIPT.turns[2].expect_user_text ?? "";
// How soon an idle tab shows a turn another tab ended: the server's word
// comes as the turn ends, and the tab then reads the chat once.
const CATCH_UP_MS = 15_000;
// Three model turns and the cost check, each of which can stall on a busy
// runner.
const TIMEOUT_MS = 120_000;

test.describe("copilot two tabs journey", () => {
  test.beforeAll(requireScriptedModel);

  test("a chat open in two tabs shows every turn in both, in order, without a reload", async ({
    page,
    context,
    journeyUser,
  }) => {
    test.setTimeout(TIMEOUT_MS);
    const run = scriptedRun("two-tabs");
    const firstMessage = `${FIRST} ${run.marker}`;
    const firstTab = await routeLiveConnection(page, { hearsTurnEnds: true });
    await startChat(page, firstMessage);
    await expectAnswer(page, FIRST_ANSWER, "the first answer should show");
    const sessionID = await currentSessionID(page);
    await expectChatSettled(page, sessionID);

    const second = await context.newPage();
    const secondTab = await routeLiveConnection(second, {
      hearsTurnEnds: false,
    });
    await second.goto(`/home?sessionId=${sessionID}`);
    await expectAnswer(
      second,
      FIRST_ANSWER,
      "the second tab should open on the chat",
    );
    await expect
      .poll(() => secondTab.signedIn(), {
        message: "the second tab's live connection should sign in",
      })
      .toBe(true);

    await sendFrom(page, SECOND);
    await expectAnswer(page, SECOND_ANSWER, "the second answer should show");
    await expectChatSettled(page, sessionID);
    await expect
      .poll(() => secondTab.turnEnds(), {
        message:
          "the server should tell the second tab that the first tab's turn ended",
      })
      .toBeGreaterThan(0);
    await expect(
      second.getByRole("log").getByText(SECOND_ANSWER, { exact: true }),
      "the second tab, never told, should still lack the first tab's turn",
    ).toHaveCount(0);
    await expect
      .poll(() => firstTab.signedIn(), {
        message: "the first tab's live connection should be signed in",
      })
      .toBe(true);

    await sendFrom(second, THIRD);
    await expectAnswer(second, THIRD_ANSWER, "the third answer should show");
    await expectChatSettled(page, sessionID);
    expect(
      await savedConversation(page, sessionID),
      "the chat should save every turn once, in the order it was sent",
    ).toEqual([
      ["user", firstMessage],
      ["assistant", FIRST_ANSWER],
      ["user", SECOND],
      ["assistant", SECOND_ANSWER],
      ["user", THIRD],
      ["assistant", THIRD_ANSWER],
    ]);
    const conversation = [
      FIRST,
      FIRST_ANSWER,
      SECOND,
      SECOND_ANSWER,
      THIRD,
      THIRD_ANSWER,
    ];
    await expect(async () => {
      await expectOnceEachInOrder(
        second,
        conversation,
        "the second tab should show the first tab's turn above its own",
      );
    }).toPass({ timeout: 10_000 });

    await expect(async () => {
      await expectOnceEachInOrder(
        page,
        conversation,
        "the first tab, idle, should show the second tab's turn " +
          `(it was told of ${firstTab.turnEnds()} turn ends of the 3)`,
      );
    }).toPass({ timeout: CATCH_UP_MS });

    expectEachTurnOnce(await readJournal(run.nonce), SCRIPT);
    if (!journeyUser.id) throw new Error("the journey user should have an id");
    await expectChargedWhatWasBilled(journeyUser.id, [run.nonce]);
  });
});

async function expectAnswer(page: Page, answer: string, message: string) {
  await expect(
    page.getByRole("log").getByText(answer, { exact: true }),
    message,
  ).toBeVisible({ timeout: 60_000 });
}

/**
 * Sends `text` from the page's composer as a new message: the composer must
 * offer Send, not Stop or Queue. It drops an Enter it takes before it is
 * ready and leaves the text in place, so Enter is pressed again while the
 * text is there and nothing has been sent.
 */
async function sendFrom(page: Page, text: string) {
  const input = page.getByRole("textbox", { name: "Chat message input" });
  await expect(input).toBeEnabled();
  await input.fill(text);
  await expect(
    page
      .locator("form")
      .filter({ has: input })
      .getByRole("button", { name: "Submit", exact: true }),
    "the composer should take a new message",
  ).toBeEnabled();
  let sent = false;
  const request = page.waitForRequest(isChatStream("POST")).then(() => {
    sent = true;
  });
  void request.catch(() => undefined);
  await expect(async () => {
    if (!sent && (await input.inputValue()) === text)
      await input.press("Enter");
    expect(sent, "the composer should send the message").toBe(true);
  }).toPass({ timeout: 15_000 });
}

/**
 * Routes the page's live connection through the journey, which counts the
 * turn ends the server sends it, and keeps them from a page that should not
 * hear them.
 */
async function routeLiveConnection(
  page: Page,
  { hearsTurnEnds }: { hearsTurnEnds: boolean },
) {
  let answered = false;
  let turnEnds = 0;
  await page.routeWebSocket(/\/ws(\?|$)/, (client) => {
    const server = client.connectToServer();
    client.onMessage((message) => server.send(message));
    server.onMessage((message) => {
      // The first answer, a pong, confirms the page's sign-in message.
      answered = true;
      if (isTurnEnd(message)) {
        turnEnds += 1;
        if (!hearsTurnEnds) return;
      }
      client.send(message);
    });
  });
  return { signedIn: () => answered, turnEnds: () => turnEnds };
}

/** The notice the server sends every tab of the user when a chat's turn ends. */
function isTurnEnd(message: string | Buffer) {
  if (typeof message !== "string") return false;
  try {
    const parsed = JSON.parse(message) as {
      method?: string;
      data?: { type?: string; event?: string };
    };
    return (
      parsed.method === "notification" &&
      parsed.data?.type === "copilot_completion" &&
      parsed.data.event === "session_completed"
    );
  } catch {
    return false;
  }
}
