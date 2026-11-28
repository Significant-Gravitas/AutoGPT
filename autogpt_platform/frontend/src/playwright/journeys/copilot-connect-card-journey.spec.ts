import {
  askForTheLine,
  connectButtons,
  connectCards,
  connectKeyOnTheCard,
  expectKeySaved,
  FIRST_MESSAGE,
  fillKeyForm,
  isCredentialsList,
  MODEL_TURN_TIMEOUT,
  SERVICE,
  savedConversation,
  savedKeys,
  toolResultTypes,
  userID,
} from "./connect-card";
import {
  chatStreamCount,
  expectChatSettled,
  finishedChatStreams,
  recordChatStreams,
} from "./copilot-chat";
import {
  deleteCredential,
  expectListedInIntegrations,
  fakeAPIKey,
  openIntegrations,
} from "./credentials";
import { expect, test } from "./fixtures";
import { pinKnownFailure } from "./known-failure";
import connectScript from "./model-scripts/connect-card-api-key.json";
import cancelScript from "./model-scripts/connect-card-cancelled.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  readJournalForKey,
  requireScriptedModel,
  scriptedRun,
  streamedTexts,
} from "./scripted-model";

// The journeys stack points Anthropic clients at the scripted model, which
// journals which key each request presented (see connect-card.ts).
const CONNECT: ModelScript = connectScript;
const CANCEL: ModelScript = cancelScript;
// What the card sends once the key is saved (SetupRequirementsCard helpers).
const CONNECTED = CONNECT.turns[2].expect_user_text ?? "";
const SKIP = CANCEL.turns[2].expect_user_text ?? "";
const BLOCK_MODEL = "claude-haiku-4-5-20251001";
const STILL_WAITING = "Once these apps are connected, Otto will continue.";
// Two model turns, a reload and the Integrations page.
const TWO_TURN_TEST_TIMEOUT = 150_000;
// A known failure must end as a failed assertion, not a timeout, so this
// stays well under the test's budget.
const REDO_CHECK_TIMEOUT = 5_000;

test.describe("copilot connect card journey", () => {
  test.beforeAll(requireScriptedModel);

  test("Otto asks for an API key on one connect card, the user adds it there, and the chat carries on by itself and runs the step once with that key", async ({
    page,
    journeyUser,
  }) => {
    test.setTimeout(TWO_TURN_TEST_TIMEOUT);
    const run = scriptedRun("connect-card-api-key");
    const newKey = { key: fakeAPIKey(), keyName: `Journey key ${run.nonce}` };
    const texts = CONNECT.turns.map((turn) => turn.text ?? "");

    await recordChatStreams(page);
    const sessionID = await askForTheLine(page, run.marker, CONNECT);
    const save = await connectKeyOnTheCard(page, newKey);
    try {
      const credential = await expectKeySaved(save, newKey);
      const transcript = page.getByRole("log");
      await expect(
        transcript.getByText(texts[3], { exact: true }),
        "the chat should carry on by itself and finish, once",
      ).toHaveCount(1, { timeout: MODEL_TURN_TIMEOUT });
      await expect(
        connectCards(transcript),
        "the card should go once it has sent its follow-up",
      ).toHaveCount(0);
      const streamed = (await finishedChatStreams(page, "POST", 2)).flatMap(
        streamedTexts,
      );
      expect(streamed, "the streams should carry what the model said").toEqual(
        texts,
      );
      await expectChatSettled(page, sessionID);
      expect(
        await chatStreamCount(page, "POST"),
        "the card should send one follow-up, and nothing else should",
      ).toBe(2);

      expect(
        await savedKeys(page),
        "the key should be saved once, as the user named it",
      ).toEqual([[newKey.keyName, "api_key"]]);
      expect(
        (await readJournalForKey(newKey.key)).map((call) => [
          call.kind,
          call.model,
          call.completed,
        ]),
        "the step should call the provider once, with the key the user saved",
      ).toEqual([["side_call", BLOCK_MODEL, true]]);
      expect(
        await toolResultTypes(page, sessionID),
        "the step should run once more after the card, not twice",
      ).toEqual(["setup_requirements", "block_output"]);
      await expect
        .poll(() => savedConversation(page, sessionID), {
          message: "the chat should save what was said, once each, in order",
        })
        .toEqual([
          ["user", `${FIRST_MESSAGE} ${run.marker}`],
          ["assistant", texts[0]],
          ["assistant", texts[1]],
          ["user", CONNECTED],
          ["assistant", texts[2]],
          ["assistant", texts[3]],
        ]);

      await page.reload();
      for (const text of [...streamed, CONNECTED]) {
        await expect(
          page.getByRole("log").getByText(text, { exact: true }),
          "the reloaded chat should show what was streamed, once",
        ).toHaveCount(1);
      }

      await openIntegrations(page);
      await expectListedInIntegrations(page, credential, SERVICE);

      const journal = await readJournal(run.nonce);
      expectEachTurnOnce(journal, CONNECT);
      await expectChargedWhatWasBilled(userID(journeyUser), [run.nonce]);
    } finally {
      if (save.credential) await deleteCredential(page, save.credential);
    }
  });

  test(
    "after a reload, nothing offers to run the finished step again, and its connect card stays gone",
    {
      annotation: {
        type: "issue",
        description:
          "https://github.com/Significant-Gravitas/AutoGPT/issues/17253",
      },
    },
    async ({ page, journeyUser: _user }) => {
      test.setTimeout(TWO_TURN_TEST_TIMEOUT);
      const run = scriptedRun("connect-card-api-key");
      const atTheCard = CONNECT.turns[1].text ?? "";
      const finished = CONNECT.turns[3].text ?? "";
      const newKey = {
        key: fakeAPIKey(),
        keyName: `Journey key ${run.nonce}`,
      };

      const sessionID = await askForTheLine(page, run.marker, CONNECT);
      const save = await connectKeyOnTheCard(page, newKey);
      try {
        await expectKeySaved(save, newKey);
        await expect(
          page.getByRole("log").getByText(finished, { exact: true }),
          "the chat should carry on by itself and finish",
        ).toHaveCount(1, { timeout: MODEL_TURN_TIMEOUT });
        await expectChatSettled(page, sessionID);

        // The card's row picks the saved key once the key list has loaded.
        const keysLoaded = page.waitForResponse(isCredentialsList);
        await page.reload();
        await keysLoaded;
        const reloaded = page.getByRole("log");
        // Settled first, the card's own answer included, so an absent card
        // means it stayed gone rather than that it hasn't drawn yet.
        await expect(
          reloaded.getByText(atTheCard, { exact: true }),
        ).toHaveCount(1);
        await expect(reloaded.getByText(finished, { exact: true })).toHaveCount(
          1,
        );
        await expect(
          page.getByRole("textbox", { name: "Chat message input" }),
        ).toBeEnabled();
        // The row names the saved key once it has picked it, which is when a
        // Proceed would show if the finished step offered one.
        await expect(
          reloaded.getByText(newKey.keyName, { exact: true }),
          "the reloaded card's row should name the saved key",
        ).toBeVisible();
        await expect(
          reloaded.getByRole("button", { name: /^(Connect|Proceed)$/ }),
          "nothing should offer to run the finished step again",
        ).toHaveCount(0, { timeout: REDO_CHECK_TIMEOUT });
        // Whether a finished card shows at all is still open (#17253): a
        // known failure on this check alone, so anything above it, and any
        // other failure after it, still fails the run.
        const pinned = pinKnownFailure(
          "#17253: the finished step's card comes back",
          "the finished step's connect card should not come back after a reload",
        );
        await expect(connectCards(reloaded), pinned).toHaveCount(0, {
          timeout: REDO_CHECK_TIMEOUT,
        });
      } finally {
        if (save.credential) await deleteCredential(page, save.credential);
      }
    },
  );

  test("cancelling the key on the card saves nothing and sends nothing: the chat stays where Otto said it needs the key, nothing is left running, and the next message is answered", async ({
    page,
    journeyUser,
  }) => {
    test.setTimeout(TWO_TURN_TEST_TIMEOUT);
    const run = scriptedRun("connect-card-cancelled");
    const texts = CANCEL.turns.map((turn) => turn.text ?? "");

    await recordChatStreams(page);
    const sessionID = await askForTheLine(page, run.marker, CANCEL);
    const transcript = page.getByRole("log");

    await connectButtons(transcript).click();
    const dialog = page.getByRole("dialog");
    await fillKeyForm(dialog, {
      key: fakeAPIKey(),
      keyName: `Journey key ${run.nonce}`,
    });
    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(dialog, "Cancel should close the dialog").toBeHidden();

    await expect(
      connectCards(transcript),
      "the card should still ask for the account",
    ).toHaveCount(1);
    await expect(
      transcript.getByText("0 of 1 ready", { exact: true }),
      "nothing should be connected",
    ).toBeVisible();
    await expect(
      transcript.getByText(STILL_WAITING, { exact: true }),
      "the card should say the chat can't go on without the account",
    ).toBeVisible();
    await expect(
      connectButtons(transcript),
      "the card should still offer to connect",
    ).toBeVisible();
    expect(
      await savedKeys(page),
      "a cancelled key should not be saved",
    ).toEqual([]);
    await expectChatSettled(page, sessionID);

    const input = page.getByRole("textbox", { name: "Chat message input" });
    await input.fill(SKIP);
    await input.press("Enter");
    await expect(
      transcript.getByText(texts[2], { exact: true }),
      "the chat should answer the user's next message, once",
    ).toHaveCount(1, { timeout: MODEL_TURN_TIMEOUT });
    await expectChatSettled(page, sessionID);
    await expect
      .poll(() => savedConversation(page, sessionID), {
        message:
          "the chat should hold only what the user and Otto said: no follow-up from the card",
      })
      .toEqual([
        ["user", `${FIRST_MESSAGE} ${run.marker}`],
        ["assistant", texts[0]],
        ["assistant", texts[1]],
        ["user", SKIP],
        ["assistant", texts[2]],
      ]);
    expect(
      await chatStreamCount(page, "POST"),
      "only the user's own messages should be sent",
    ).toBe(2);
    expect(
      await toolResultTypes(page, sessionID),
      "the step should not run again",
    ).toEqual(["setup_requirements"]);

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, CANCEL);
    await expectChargedWhatWasBilled(userID(journeyUser), [run.nonce]);
  });
});
