import { describe, expect, it } from "vitest";
import { backgroundRefreshInterval } from "../backgroundSubtasks";
import { convertChatSessionMessagesToUiMessages } from "../convertChatSessionToUiMessages";

const accepted = {
  role: "tool",
  content: JSON.stringify({
    background_task_id: "task",
    background_expires_at: "2026-10-19T12:10:00Z",
  }),
};
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
  it.each([undefined, null, {}, "unavailable"])(
    "does not poll absent message history (%s)",
    (messages) => {
      expect(backgroundRefreshInterval(messages, now)).toBe(false);
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

  it("keeps checking after the original reply and through a queued wake", () => {
    expect(
      backgroundRefreshInterval(
        [accepted, { role: "assistant", content: "I'll follow up." }],
        now,
      ),
    ).toBe(3000);
    expect(backgroundRefreshInterval([accepted, wake], now)).toBe(3000);
    expect(
      backgroundRefreshInterval(
        [accepted, wake, { role: "assistant", content: "Here is the result." }],
        now,
      ),
    ).toBe(false);
  });

  it.each(["expert_run", "scheduled_chat_notice"])(
    "keeps checking when a standalone %s arrives before the follow-up",
    (kind) => {
      const messages = [
        accepted,
        wake,
        {
          role: "assistant",
          content: "An unrelated update.",
          metadata: { kind },
        },
      ];
      expect(backgroundRefreshInterval(messages, now)).toBe(3000);
      expect(
        backgroundRefreshInterval(
          [...messages, { role: "assistant", content: "Here is the result." }],
          now,
        ),
      ).toBe(false);
    },
  );

  it("does not poll foreground chats or expired background requests", () => {
    expect(
      backgroundRefreshInterval(
        [{ role: "tool", content: '{"status":"running"}' }],
        now,
      ),
    ).toBe(false);
    expect(backgroundRefreshInterval([accepted], now + 11 * 60_000)).toBe(
      false,
    );
    expect(
      backgroundRefreshInterval(
        [
          accepted,
          {
            role: "assistant",
            metadata: { background_task_notice: "task" },
            content: "Follow-up expired.",
          },
        ],
        now,
      ),
    ).toBe(false);
  });

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
