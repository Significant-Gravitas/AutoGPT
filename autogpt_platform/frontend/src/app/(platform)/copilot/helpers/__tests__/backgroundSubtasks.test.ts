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

describe("background subtask delivery", () => {
  it.each([undefined, null, {}, { background_pending: false }])(
    "does not poll a chat with nothing pending (%s)",
    (session) => {
      expect(backgroundRefreshInterval(session)).toBe(false);
    },
  );

  it("keeps checking while the server reports a continuation still to come", () => {
    expect(backgroundRefreshInterval({ background_pending: true })).toBe(3000);
  });

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
