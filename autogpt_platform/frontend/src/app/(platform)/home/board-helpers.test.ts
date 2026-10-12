import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { describe, expect, it } from "vitest";
import {
  type Card,
  cardState,
  dropPosition,
  groupCards,
  parseMention,
  positionBetween,
  stateLineText,
} from "./board-helpers";

const NOW = new Date("2026-10-12T12:00:00Z");
const DAY_MS = 24 * 60 * 60 * 1000;

const COLUMNS: BoardColumn[] = [
  { key: "backlog", name: "Backlog", builtin: true },
  { key: "in_progress", name: "In progress", builtin: true },
  { key: "done", name: "Done", builtin: true },
];

function card(id: string, overrides: Partial<Card> = {}): Card {
  return {
    id,
    created_at: NOW.toISOString(),
    updated_at: NOW.toISOString(),
    is_processing: false,
    ...overrides,
  };
}

function daysAgo(days: number) {
  return new Date(NOW.getTime() - days * DAY_MS).toISOString();
}

function idsByColumn(columns: ReturnType<typeof groupCards>) {
  return Object.fromEntries(
    columns.map(({ column, cards }) => [column.key, cards.map((c) => c.id)]),
  );
}

describe("groupCards", () => {
  it("puts placed cards in their column sorted by position", () => {
    const grouped = groupCards(
      COLUMNS,
      [
        card("b2", { board_column: "backlog", board_position: 2048 }),
        card("b1", { board_column: "backlog", board_position: 1024 }),
        card("d1", { board_column: "done", board_position: 0 }),
      ],
      NOW,
    );

    expect(idsByColumn(grouped)).toEqual({
      backlog: ["b1", "b2"],
      in_progress: [],
      done: ["d1"],
    });
  });

  it("puts recent unplaced cards in In progress after placed ones, newest first", () => {
    const grouped = groupCards(
      COLUMNS,
      [
        card("old-recent", { updated_at: daysAgo(3) }),
        card("placed", { board_column: "in_progress", board_position: 5 }),
        card("new-recent", { updated_at: daysAgo(1) }),
      ],
      NOW,
    );

    expect(idsByColumn(grouped).in_progress).toEqual([
      "placed",
      "new-recent",
      "old-recent",
    ]);
  });

  it("puts a card with an unknown column in Backlog", () => {
    const grouped = groupCards(
      COLUMNS,
      [card("lost", { board_column: "col_gone", board_position: 0 })],
      NOW,
    );

    expect(idsByColumn(grouped).backlog).toEqual(["lost"]);
  });

  it("puts older unplaced cards in Done after placed ones", () => {
    const grouped = groupCards(
      COLUMNS,
      [
        card("ancient", { updated_at: daysAgo(30) }),
        card("finished", { board_column: "done", board_position: 0 }),
        card("older", { updated_at: daysAgo(10) }),
      ],
      NOW,
    );

    expect(idsByColumn(grouped).done).toEqual(["finished", "older", "ancient"]);
    expect(idsByColumn(grouped).in_progress).toEqual([]);
  });

  it("keeps custom columns in the order the board returns them", () => {
    const grouped = groupCards(
      [...COLUMNS, { key: "col_review", name: "Review" }],
      [card("r", { board_column: "col_review", board_position: 0 })],
      NOW,
    );

    expect(grouped.map(({ column }) => column.key)).toEqual([
      "backlog",
      "in_progress",
      "done",
      "col_review",
    ]);
    expect(idsByColumn(grouped).col_review).toEqual(["r"]);
  });
});

describe("positionBetween", () => {
  it("returns 0 for an empty column", () => {
    expect(positionBetween(null, null)).toBe(0);
  });

  it("goes before the first card", () => {
    expect(positionBetween(null, 100)).toBe(100 - 1024);
  });

  it("goes after the last card", () => {
    expect(positionBetween(100, null)).toBe(100 + 1024);
  });

  it("goes halfway between two cards", () => {
    expect(positionBetween(100, 200)).toBe(150);
  });
});

describe("dropPosition", () => {
  const column = [
    card("a", { board_position: 1000 }),
    card("b", { board_position: 2000 }),
    card("c", { board_position: 3000 }),
  ];

  it("drops at the top", () => {
    expect(dropPosition(column, 0, "x")).toBe(1000 - 1024);
  });

  it("drops at the bottom", () => {
    expect(dropPosition(column, 3, "x")).toBe(3000 + 1024);
  });

  it("drops in the middle", () => {
    expect(dropPosition(column, 1, "x")).toBe(1500);
  });

  it("ignores the moving card when it stays in the same column", () => {
    // "a" moves below "b": without "a" the list is [b, c], index 1 sits between them.
    expect(dropPosition(column, 1, "a")).toBe(2500);
  });

  it("treats unplaced neighbours as absent", () => {
    const mixed = [
      card("placed", { board_position: 1000 }),
      card("unplaced", { board_position: null }),
    ];
    expect(dropPosition(mixed, 1, "x")).toBe(1000 + 1024);
    expect(dropPosition(mixed, 0, "x")).toBe(1000 - 1024);
  });
});

describe("cardState", () => {
  it("prefers running over everything", () => {
    const state = cardState(
      card("x", {
        is_processing: true,
        needs_you: true,
        started: false,
        last_line: "hi",
      }),
    );
    expect(state).toEqual({ kind: "running" });
    expect(stateLineText(state)).toBe("Working…");
  });

  it("then needs you", () => {
    const state = cardState(
      card("x", { needs_you: true, started: true, last_line: "hi" }),
    );
    expect(state).toEqual({ kind: "needs_you" });
    expect(stateLineText(state)).toBe("Waiting for your answer");
  });

  it("then not started", () => {
    const state = cardState(card("x", { started: false, last_line: "hi" }));
    expect(state).toEqual({ kind: "not_started" });
    expect(stateLineText(state)).toBe("Not started");
  });

  it("then the last line", () => {
    const state = cardState(card("x", { started: true, last_line: "Done." }));
    expect(state).toEqual({ kind: "line", text: "Done." });
    expect(stateLineText(state)).toBe("Done.");
  });

  it("falls back to empty", () => {
    const state = cardState(card("x", { started: true, last_line: null }));
    expect(state).toEqual({ kind: "empty" });
    expect(stateLineText(state)).toBeNull();
  });
});

describe("parseMention", () => {
  it("opens on a trailing @ after a space and strips it", () => {
    expect(parseMention("fix bug @")).toEqual({
      text: "fix bug ",
      mentionOpen: true,
    });
  });

  it("does not open on an @ inside a word", () => {
    expect(parseMention("a@b")).toEqual({ text: "a@b", mentionOpen: false });
    expect(parseMention("mail a@")).toEqual({
      text: "mail a@",
      mentionOpen: false,
    });
  });

  it("opens on a lone @", () => {
    expect(parseMention("@")).toEqual({ text: "", mentionOpen: true });
  });
});
