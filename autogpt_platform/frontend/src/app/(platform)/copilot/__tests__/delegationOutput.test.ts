import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { describe, expect, it } from "vitest";
import {
  applyHeldOutcome,
  applyOutput,
  isTurnedDown,
} from "../delegationOutput";
import { askedQuestionOf, pendingQuestionOf } from "../delegationLiveStatus";
import { makeDelegation } from "./delegationFixture";

const PART = { type: "tool-delegate_to_expert", state: "output-available" };

describe("applyOutput", () => {
  it("fails with the stream's own error text", () => {
    const failed = applyOutput(
      makeDelegation(),
      { type: PART.type, state: "output-error", errorText: "Tool crashed" },
      null,
    );
    expect(failed).toMatchObject({ status: "failed", error: "Tool crashed" });
  });

  it("reads files with sizes and the options key", () => {
    const output = {
      status: "needs_input",
      question: "Which?",
      options: ["A", "B"],
      sub_workspace_files: [
        { name: "a.md", path: "/a", size: 10 },
        { name: "b.md", path: "/b" },
        { name: "no-path" },
        "junk",
      ],
    };
    const out = applyOutput(makeDelegation(), { ...PART, output }, output);
    expect(out.questionOptions).toEqual(["A", "B"]);
    expect(out.files).toEqual([
      { name: "a.md", path: "/a", sizeBytes: 10 },
      { name: "b.md", path: "/b", sizeBytes: null },
    ]);
  });

  it("keeps the server's reason for a cancelled run", () => {
    const output = { status: "cancelled", error: "Parent turn stopped" };
    const out = applyOutput(makeDelegation(), { ...PART, output }, output);
    expect(out).toMatchObject({
      status: "cancelled",
      error: "Parent turn stopped",
    });
  });
});

describe("applyHeldOutcome", () => {
  const held = makeDelegation({ status: "proposed", subSessionId: null });

  it("reads an expired approval as turned down", () => {
    const out = applyHeldOutcome(held, { outcome: "expired", output: "" });
    expect(out).toMatchObject({
      status: "cancelled",
      error: "The approval expired",
    });
    expect(isTurnedDown(out)).toBe(true);
  });

  it("takes a plain-text result as the answer, and none as stopped", () => {
    expect(
      applyHeldOutcome(held, { outcome: "approved", output: "Done, see file" }),
    ).toMatchObject({
      status: "completed",
      response: "Done, see file",
      approved: true,
    });
    const stopped = applyHeldOutcome(held, { outcome: "approved", output: "" });
    expect(stopped).toMatchObject({ status: "cancelled", error: "Stopped" });
    expect(isTurnedDown(stopped)).toBe(false);
  });
});

describe("question readers", () => {
  it("reads the pending question and when it was asked", () => {
    const session = {
      metadata: {
        pending_question: { text: "Q4?", asked_at: "2026-09-28T10:03:00Z" },
      },
    } as unknown as SessionDetailResponse;
    expect(pendingQuestionOf(session)).toEqual({
      text: "Q4?",
      askedAt: Date.parse("2026-09-28T10:03:00Z"),
    });
    expect(
      pendingQuestionOf({
        metadata: { pending_question: { text: "Q?", asked_at: "nope" } },
      } as unknown as SessionDetailResponse)?.askedAt,
    ).toBeNull();
    expect(pendingQuestionOf({} as SessionDetailResponse)).toBeNull();
  });

  it("joins several questions and tolerates odd input", () => {
    expect(
      askedQuestionOf([
        {
          name: "ask_question",
          input: {
            questions: [{ question: "A?" }, "junk", { question: "B?" }],
          },
        },
      ]),
    ).toEqual({ text: "A? B?", options: [] });
    expect(askedQuestionOf([{ name: "ask_question", input: null }])).toEqual({
      text: null,
      options: [],
    });
  });
});
