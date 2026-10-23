import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import { getVisibleUserMessageParts } from "../userMessageParts";

const delegation =
  "[Delegated task from Avery, a teammate on this user's team — not " +
  "the user. They cannot see your thread, so report the outcome in your " +
  "final message. If the task needs something only the user can " +
  "provide, say what is missing instead of guessing.]";

const handoff =
  "[Task handed to you by Avery, a teammate on this user's team. It " +
  "is yours now: they are not waiting on a report and cannot answer " +
  "follow-ups. Take it to completion and tell the user the outcome " +
  "yourself. If something only the user can provide is missing, ask " +
  "them.]";

describe("getVisibleUserMessageParts", () => {
  it.each([delegation, handoff])(
    "removes the generated preamble while preserving task content: %s",
    (preamble) => {
      const text = `${preamble}\n\n[Context: Launch on Friday.]\n\nReview **all** links.\n\n- [ ] Footer`;
      const parts: UIMessage["parts"] = [{ type: "text", text }];

      expect(getVisibleUserMessageParts(parts)).toEqual([
        {
          type: "text",
          text: "[Context: Launch on Friday.]\n\nReview **all** links.\n\n- [ ] Footer",
        },
      ]);
      expect(parts[0]).toEqual({ type: "text", text });
    },
  );

  it("preserves attachments and later text parts verbatim", () => {
    const attachment = {
      type: "file" as const,
      mediaType: "image/png",
      url: "https://example.com/brief.png",
    };
    const quotedPreamble = {
      type: "text" as const,
      text: `${delegation}\n\nA quoted example within the task.`,
    };
    const parts: UIMessage["parts"] = [
      attachment,
      { type: "text", text: `${delegation}\n\nReview the attached brief.` },
      quotedPreamble,
    ];

    expect(getVisibleUserMessageParts(parts)).toEqual([
      attachment,
      { type: "text", text: "Review the attached brief." },
      quotedPreamble,
    ]);
  });

  it.each([
    "An ordinary message.",
    "[Delegated task from Avery]\n\nKeep this user-written note.",
    "[Context: User-written background.]\n\nKeep this too.",
    `Please explain this wrapper:\n\n${delegation}\n\nTask`,
    `${delegation}\n\n`,
  ])("preserves ordinary, quoted and incomplete content: %s", (text) => {
    const parts: UIMessage["parts"] = [{ type: "text", text }];

    expect(getVisibleUserMessageParts(parts)).toBe(parts);
  });

  it("preserves a message without text", () => {
    const parts: UIMessage["parts"] = [];

    expect(getVisibleUserMessageParts(parts)).toBe(parts);
  });
});
