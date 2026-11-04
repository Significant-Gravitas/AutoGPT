import { describe, expect, it } from "vitest";
import { backgroundRefreshInterval } from "../backgroundSubtasks";
import { convertChatSessionMessagesToUiMessages } from "../convertChatSessionToUiMessages";

const wake = {
  role: "user",
  content: "Internal prompt containing untrusted subtask output",
  sequence: 3,
  metadata: {
    background_task_notice: "task",
    background_task_stage: "followup",
    display_content: "Background subtask completed. Otto is following up.",
  },
};
const now = Date.parse("2026-10-19T12:00:00Z");

describe("background subtask delivery", () => {
  it.each([undefined, null, {}, { background_pending_until: null }])(
    "does not poll a chat with nothing pending (%s)",
    (session) => {
      expect(backgroundRefreshInterval(session, now)).toBe(false);
    },
  );

  it.each([undefined, null, 42, ""])(
    "hides the internal prompt when the display label is unusable (%s)",
    (display_content) => {
      const result = convertChatSessionMessagesToUiMessages(
        "session",
        [{ ...wake, metadata: { ...wake.metadata, display_content } }],
        { isComplete: true },
      );
      expect(result.messages[0].role).toBe("assistant");
      expect(JSON.stringify(result.messages)).not.toContain("Internal prompt");
      expect(JSON.stringify(result.messages)).toContain(
        "Otto is following up.",
      );
    },
  );

  it.each([
    "2026-10-19T12:10:00Z",
    "2026-10-19T12:10:00+00:00",
    new Date("2026-10-19T12:10:00Z"),
  ])(
    "keeps checking until the follow-up window the server reports has passed (%s)",
    (background_pending_until) => {
      const session = { background_pending_until };
      expect(backgroundRefreshInterval(session, now)).toBe(3000);
      expect(backgroundRefreshInterval(session, now + 11 * 60_000)).toBe(false);
    },
  );

  it("labels an automatic wake as Otto's update without exposing its internal prompt", () => {
    const result = convertChatSessionMessagesToUiMessages(
      "session",
      [
        { role: "assistant", content: "Earlier reply", sequence: 2 },
        wake,
        { role: "assistant", content: "The answer is 42.", sequence: 4 },
      ],
      { isComplete: true },
    );
    expect(result.messages).toHaveLength(2);
    expect(result.messages[1].role).toBe("assistant");
    expect(JSON.stringify(result.messages)).not.toContain("Internal prompt");
    expect(JSON.stringify(result.messages[1].parts)).toContain(
      "Background subtask completed",
    );
    expect(JSON.stringify(result.messages[1].parts)).toContain(
      "The answer is 42.",
    );
  });
});
