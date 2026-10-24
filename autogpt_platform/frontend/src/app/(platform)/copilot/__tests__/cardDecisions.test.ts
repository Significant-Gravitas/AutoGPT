import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import {
  type Decision,
  decisionLine,
  decisionNotes,
  getAnsweredCards,
  getCardDecisions,
} from "../cardDecisions";
import cases from "./cardDecisionLines.json";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// The same file pins gate/card_approval.py's _DECISION to these lines
// (card_decision_lines_test.py), so the card and the gate read one wording.
const CASES = cases as {
  kind: "hire" | "raise" | "update" | "soul" | null;
  name: string;
  confirmation_id: string;
  verdict: Decision;
  verb: string;
  line: string;
  note: string;
}[];

function userRow(text: string, metadata?: unknown): Message {
  return {
    id: "u1",
    role: "user",
    ...(metadata ? { metadata } : {}),
    parts: [{ type: "text", text }],
  };
}

describe("card decision lines", () => {
  it.each(CASES)("the card sends $line", (c) => {
    expect(
      decisionLine({
        kind: c.kind,
        name: c.name,
        confirmationId: c.confirmation_id,
        decision: c.verdict,
      }),
    ).toBe(c.line);
  });

  it.each(CASES)("the chat reads $line as: $note", (c) => {
    const decisions = getCardDecisions(userRow(c.line));

    expect(decisions).toEqual([
      {
        verdict: c.verdict,
        verb: c.verb,
        name: c.name,
        confirmationId: c.confirmation_id,
      },
    ]);
    expect(decisionNotes(decisions ?? []).map((n) => n.text)).toEqual([c.note]);
  });

  it("prefers what the server recorded over the row's text", () => {
    const decisions = getCardDecisions(
      userRow("Approved: hire Maya (confirmation_id: c-1).", {
        kind: "card_decision",
        decisions: [
          {
            verdict: "declined",
            confirmation_id: "c-1",
            verb: "hire",
            name: "Maya",
          },
        ],
      }),
    );

    expect(decisions).toEqual([
      {
        verdict: "declined",
        verb: "hire",
        name: "Maya",
        confirmationId: "c-1",
      },
    ]);
  });

  it("joins the same answer to the same change into one sentence", () => {
    const decisions = getCardDecisions(
      userRow(
        [
          "Approved: hire Maya (confirmation_id: c-1).",
          "Approved: hire Theo (confirmation_id: c-2).",
          "Approved: hire Vera (confirmation_id: c-3).",
          "Not approved: do not hire Zara, discard that proposal (confirmation_id: c-4).",
          "Approved: go ahead (confirmation_id: c-5).",
        ].join("\n"),
      ),
    );

    expect(decisionNotes(decisions ?? [])).toEqual([
      { verdict: "approved", text: "You approved hiring Maya, Theo, and Vera" },
      { verdict: "declined", text: "You declined hiring Zara" },
      { verdict: "approved", text: "You approved a proposed change" },
    ]);
  });

  it.each([
    ["the user's own words", "Approved: hire Maya, she looks great."],
    [
      "a decision with more text",
      "Approved: hire Maya (confirmation_id: c-1).\nStart with Instagram.",
    ],
    ["nothing", ""],
  ])("leaves %s as the user's message", (_, text) => {
    expect(getCardDecisions(userRow(text))).toBeNull();
  });

  it("leaves a decision sent with an attachment as the user's message", () => {
    const withFile: Message = {
      id: "u1",
      role: "user",
      metadata: {
        kind: "card_decision",
        decisions: [{ verdict: "approved", verb: "hire", name: "Maya" }],
      },
      parts: [
        { type: "text", text: "Approved: hire Maya (confirmation_id: c-1)." },
        {
          type: "file",
          mediaType: "application/pdf",
          filename: "brief.pdf",
          url: "/api/proxy/api/workspace/files/f-1/download",
        },
      ],
    };

    expect(getCardDecisions(withFile)).toBeNull();
  });

  it("never reads Otto's own words as a decision", () => {
    const reply: Message = {
      id: "a1",
      role: "assistant",
      parts: [
        { type: "text", text: "Approved: hire Maya (confirmation_id: c-1)." },
      ],
    };

    expect(getCardDecisions(reply)).toBeNull();
  });
});

describe("answered cards", () => {
  it("knows every card the user's rows answered, the last answer standing", () => {
    const rows: Message[] = [
      { ...userRow("Approved: hire Maya (confirmation_id: c-1)."), id: "u1" },
      {
        ...userRow(
          "Approved: hire Ada (confirmation_id: c-2).\nAnd start on Monday.",
        ),
        id: "u2",
      },
      {
        ...userRow(
          "Not approved: do not hire Maya, discard that proposal (confirmation_id: c-1).",
        ),
        id: "u3",
      },
      {
        id: "a1",
        role: "assistant",
        parts: [
          { type: "text", text: "Approved: hire Bo (confirmation_id: c-3)." },
        ],
      },
    ];

    expect(getAnsweredCards(rows)).toEqual(
      new Map([
        ["c-1", { verdict: "declined", messageID: "u3" }],
        ["c-2", { verdict: "approved", messageID: "u2" }],
      ]),
    );
  });
});
