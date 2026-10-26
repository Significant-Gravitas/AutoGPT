import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import {
  type Decision,
  decisionLine,
  decisionNotes,
  getAnsweredCards,
  getCardDecisions,
} from "../cardDecisions";
import edgeLines from "./cardDecisionEdgeLines.json";
import cases from "./cardDecisionLines.json";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// The same file pins gate/card_approval.py's _DECISION to these lines
// (card_decision_lines_test.py), so the card and the gate read one wording.
const CASES = cases as {
  kind: "hire" | "raise" | "update" | "soul" | null;
  /** The name as the card shows it, when the line writes it plainer. */
  card_name?: string;
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
        name: c.card_name ?? c.name,
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

  // The gate reads the same file (card_decision_lines_test.py): a typed or
  // pasted line with odd characters is a decision on both sides or neither.
  it.each(edgeLines)("reads a line with $why as the gate does", (c) => {
    expect(getCardDecisions(userRow(c.text)) !== null).toBe(c.decides);
  });

  it("reads a long run of whitespace in linear time", () => {
    // User text of any length: the bound is ~1000x the linear cost.
    const line = "x" + "\t".repeat(50_000) + "x";

    const started = performance.now();
    getCardDecisions(userRow(line));

    expect(performance.now() - started).toBeLessThan(2000);
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
        ].join("\n"),
      ),
    );

    expect(decisionNotes(decisions ?? [])).toEqual([
      { verdict: "approved", text: "You approved hiring Maya, Theo, and Vera" },
      { verdict: "declined", text: "You declined hiring Zara" },
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

  it("leaves a row the server wrote for something else as it is", () => {
    // A subtask's prompt arrives as a user row tagged with where it came from.
    const prompt: Message = {
      ...userRow("Approved: hire Maya (confirmation_id: c-1)."),
      metadata: {
        from_session_id: "6f1c2a3e-7b1d-4c9e-8a6f-0d2b4e6c8a10",
        from_expert_id: null,
      },
    };

    expect(getCardDecisions(prompt)).toBeNull();
    expect(getAnsweredCards([prompt]).size).toBe(0);
  });

  it("reads a decision queued behind a running turn, whose row carries the queue's metadata", () => {
    const queued: Message = {
      ...userRow("Approved: hire Maya (confirmation_id: c-1)."),
      metadata: { cost_context: {}, llm_auth_provider: "platform" },
    };

    expect(getCardDecisions(queued)).toEqual([
      {
        verdict: "approved",
        verb: "hire",
        name: "Maya",
        confirmationId: "c-1",
      },
    ]);
    expect(getAnsweredCards([queued]).get("c-1")?.verdict).toBe("approved");
  });

  it.each([
    ["a held call's result", { held_call: { review_id: "r1" } }],
    ["an answered-approvals wake", { held_calls_answered: 1 }],
    ["a hidden kickoff", { hidden: true, kind: "expert_kickoff" }],
  ])("leaves %s as it is", (_, metadata) => {
    const row: Message = {
      ...userRow("Approved: hire Maya (confirmation_id: c-1)."),
      metadata,
    };

    expect(getCardDecisions(row)).toBeNull();
    expect(getAnsweredCards([row]).size).toBe(0);
  });

  it("leaves a decision in other words as the user's message", () => {
    expect(
      getCardDecisions(
        userRow("Approved: delete all my files (confirmation_id: c-1)."),
      ),
    ).toBeNull();
  });

  it("does not speak for the owner in a shared chat", () => {
    const decisions = getCardDecisions(
      userRow("Approved: hire Maya (confirmation_id: c-1)."),
    );

    expect(
      decisionNotes(decisions ?? [], { readOnly: true }).map((n) => n.text),
    ).toEqual(["Approved hiring Maya"]);
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

  it("lets a saved answer win over a refused one still on screen", () => {
    const rows: Message[] = [
      {
        ...userRow("Approved: hire Maya (confirmation_id: c-1)."),
        id: "s1-seq-4",
      },
      // A send the server refused: never saved, still shown by the client.
      {
        ...userRow(
          "Not approved: do not hire Maya, discard that proposal (confirmation_id: c-1).",
        ),
        id: "optimistic-1",
      },
    ];

    expect(getAnsweredCards(rows).get("c-1")).toEqual({
      verdict: "approved",
      messageID: "s1-seq-4",
    });
  });
});
