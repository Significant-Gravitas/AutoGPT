import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { describe, expect, it } from "vitest";
import {
  applyReassign,
  type Card,
  cardColumnKey,
  cardState,
  dragLabel,
  dropPosition,
  groupCards,
  isSessionId,
  moveCard,
  moveCardToTop,
  newColumnKey,
  nextColumnName,
  parseMention,
  patchCard,
  planDrop,
  positionBetween,
  resolveDrop,
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

  it("drops among unplaced cards right after the last placed card", () => {
    const mixed = [
      card("placed", { board_position: 1000 }),
      card("u1", { board_position: null }),
      card("u2", { board_position: null }),
    ];
    expect(dropPosition(mixed, 2, "x")).toBe(1000 + 1024);
    expect(dropPosition(mixed.slice(1), 1, "x")).toBe(0);
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

describe("cardColumnKey", () => {
  const keys = new Set(COLUMNS.map((column) => column.key));

  it("uses the placed column", () => {
    expect(cardColumnKey(card("a", { board_column: "done" }), keys, NOW)).toBe(
      "done",
    );
  });

  it("sends an unknown column to Backlog", () => {
    expect(
      cardColumnKey(card("a", { board_column: "col_gone" }), keys, NOW),
    ).toBe("backlog");
  });

  it("places unplaced cards by age", () => {
    expect(cardColumnKey(card("a"), keys, NOW)).toBe("in_progress");
    expect(
      cardColumnKey(card("a", { updated_at: daysAgo(8) }), keys, NOW),
    ).toBe("done");
  });
});

describe("patchCard / moveCard", () => {
  const board = {
    columns: COLUMNS,
    sessions: [
      card("a", { board_column: "backlog", board_position: 0 }),
      card("b", { board_column: "done", board_position: 5 }),
    ],
  };

  it("patches only the matching card and leaves the input alone", () => {
    const next = patchCard(board, "a", { title: "Renamed" });
    expect(next.sessions[0].title).toBe("Renamed");
    expect(next.sessions[1]).toBe(board.sessions[1]);
    expect(board.sessions[0].title).toBeUndefined();
  });

  it("moves a card to a column and position", () => {
    const next = moveCard(board, "a", "in_progress", 42);
    expect(next.sessions[0]).toMatchObject({
      board_column: "in_progress",
      board_position: 42,
    });
  });

  it("returns the board unchanged for an unknown card", () => {
    expect(moveCard(board, "zzz", "done", 1).sessions).toEqual(board.sessions);
  });

  it("leaves the input board and card alone", () => {
    const next = moveCard(board, "a", "done", 7);
    expect(next).not.toBe(board);
    expect(next.sessions[0]).not.toBe(board.sessions[0]);
    expect(board.sessions[0]).toMatchObject({
      board_column: "backlog",
      board_position: 0,
    });
    expect(next.columns).toBe(board.columns);
  });
});

describe("moveCardToTop", () => {
  it("goes above the column's first placed card", () => {
    const board = {
      columns: COLUMNS,
      sessions: [
        card("a", { board_column: "backlog", board_position: 0 }),
        card("b", { board_column: "done", board_position: 5 }),
        card("c", { board_column: "done", board_position: 9 }),
      ],
    };
    const next = moveCardToTop(board, "a", "done");
    expect(next.sessions[0]).toMatchObject({
      board_column: "done",
      board_position: 5 - 1024,
    });
    expect(idsByColumn(groupCards(COLUMNS, next.sessions, NOW)).done).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  it("lands at 0 in a column with no placed cards", () => {
    const board = { columns: COLUMNS, sessions: [card("a")] };
    expect(moveCardToTop(board, "a", "done").sessions[0]).toMatchObject({
      board_column: "done",
      board_position: 0,
    });
  });

  it("ignores the moving card's own position", () => {
    const board = {
      columns: COLUMNS,
      sessions: [
        card("a", { board_column: "done", board_position: 3 }),
        card("b", { board_column: "done", board_position: 7 }),
      ],
    };
    expect(moveCardToTop(board, "b", "done").sessions[1].board_position).toBe(
      3 - 1024,
    );
  });
});

describe("applyReassign", () => {
  const board = {
    columns: COLUMNS,
    sessions: [
      card("a", { board_column: "in_progress", board_position: 0 }),
      card("b", { board_column: "done", board_position: 5 }),
    ],
  };
  const oldSession = card("a", {
    board_column: "done",
    board_position: -1019,
    last_line: "Reassigned to Priya",
  });
  const newSession = card("n", {
    board_column: "in_progress",
    board_position: -1024,
    expert_id: "expert-priya",
  });

  it("moves the old card to Done and adds the new card", () => {
    const next = applyReassign(board, oldSession, newSession);
    expect(idsByColumn(groupCards(COLUMNS, next.sessions, NOW))).toEqual({
      backlog: [],
      in_progress: ["n"],
      done: ["a", "b"],
    });
    expect(next.sessions.find((c) => c.id === "a")?.last_line).toBe(
      "Reassigned to Priya",
    );
    expect(board.sessions).toHaveLength(2);
  });

  it("does not add the new card twice", () => {
    const once = applyReassign(board, oldSession, newSession);
    const twice = applyReassign(once, oldSession, newSession);
    expect(twice.sessions.filter((c) => c.id === "n")).toHaveLength(1);
  });
});

describe("isSessionId", () => {
  it("accepts a UUID in either case", () => {
    expect(isSessionId("3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b")).toBe(true);
    expect(isSessionId("3F2B8C1E-9A4D-4E6F-8B7A-1C2D3E4F5A6B")).toBe(true);
  });

  it("rejects anything that could change an API path", () => {
    expect(isSessionId(null)).toBe(false);
    expect(isSessionId("")).toBe(false);
    expect(isSessionId("../../x")).toBe(false);
    expect(isSessionId("3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b/../x")).toBe(
      false,
    );
    expect(isSessionId("s1")).toBe(false);
  });
});

describe("resolveDrop / planDrop", () => {
  const grouped = groupCards(
    COLUMNS,
    [
      card("a", { board_column: "backlog", board_position: 0 }),
      card("b", { board_column: "backlog", board_position: 1024 }),
      card("c", { board_column: "in_progress", board_position: 0 }),
      card("d", { board_column: "in_progress", board_position: 1024 }),
      card("e", { board_column: "in_progress", board_position: 2048 }),
    ],
    NOW,
  );

  it("drops over a card in another column at that card's index", () => {
    expect(resolveDrop(grouped, "a", "d")).toEqual({
      column: "in_progress",
      index: 1,
    });
  });

  it("drops over a column at its end", () => {
    expect(resolveDrop(grouped, "a", "in_progress")).toEqual({
      column: "in_progress",
      index: 3,
    });
    expect(resolveDrop(grouped, "a", "done")).toEqual({
      column: "done",
      index: 0,
    });
  });

  it("does not count the moving card when it drops over its own column", () => {
    expect(resolveDrop(grouped, "a", "backlog")).toEqual({
      column: "backlog",
      index: 1,
    });
  });

  it("drops over itself at its own index", () => {
    expect(resolveDrop(grouped, "d", "d")).toEqual({
      column: "in_progress",
      index: 1,
    });
  });

  it("returns null over nothing known", () => {
    expect(resolveDrop(grouped, "a", "nowhere")).toBeNull();
  });

  it("goes after a card in another column when dropped on its lower half", () => {
    expect(resolveDrop(grouped, "a", "d", true)).toEqual({
      column: "in_progress",
      index: 2,
    });
    expect(resolveDrop(grouped, "a", "d", false)).toEqual({
      column: "in_progress",
      index: 1,
    });
  });

  it("reaches the end of another column below its last card", () => {
    expect(resolveDrop(grouped, "a", "e", true)).toEqual({
      column: "in_progress",
      index: 3,
    });
    expect(planDrop(grouped, "a", "e", true)).toEqual({
      column: "in_progress",
      position: 2048 + 1024,
    });
  });

  it("keeps the over card's index within the same column", () => {
    expect(resolveDrop(grouped, "c", "d", true)).toEqual({
      column: "in_progress",
      index: 1,
    });
    expect(resolveDrop(grouped, "e", "d", true)).toEqual({
      column: "in_progress",
      index: 1,
    });
  });

  it("plans a move between the cards around the drop", () => {
    expect(planDrop(grouped, "a", "d")).toEqual({
      column: "in_progress",
      position: 512,
    });
    expect(planDrop(grouped, "a", "in_progress")).toEqual({
      column: "in_progress",
      position: 2048 + 1024,
    });
  });

  it("moves down within a column past the card it drops on", () => {
    expect(planDrop(grouped, "c", "e")).toEqual({
      column: "in_progress",
      position: 2048 + 1024,
    });
  });

  it("plans nothing when the card lands where it was", () => {
    expect(planDrop(grouped, "d", "d")).toBeNull();
    expect(planDrop(grouped, "b", "backlog")).toBeNull();
    expect(planDrop(grouped, "a", "nowhere")).toBeNull();
    expect(planDrop(grouped, "zzz", "d")).toBeNull();
  });
});

describe("dragLabel", () => {
  const grouped = groupCards(
    COLUMNS,
    [
      card("a", { board_column: "backlog", title: "Write PRD" }),
      card("b", { board_column: "backlog" }),
    ],
    NOW,
  );

  it("names a column, a card or neither", () => {
    expect(dragLabel(grouped, "done")).toBe("Done");
    expect(dragLabel(grouped, "a")).toBe("Write PRD");
    expect(dragLabel(grouped, "b")).toBe("Untitled task");
    expect(dragLabel(grouped, "zzz")).toBe("the board");
  });
});

describe("nextColumnName", () => {
  it("names the first new column plainly", () => {
    expect(nextColumnName(COLUMNS)).toBe("New column");
  });

  it("numbers the next free name", () => {
    expect(
      nextColumnName([
        ...COLUMNS,
        { key: "col_1", name: "New column" },
        { key: "col_2", name: "New column 2" },
      ]),
    ).toBe("New column 3");
    expect(
      nextColumnName([...COLUMNS, { key: "col_2", name: "New column 2" }]),
    ).toBe("New column");
  });
});

describe("newColumnKey", () => {
  it("makes a unique col_<uuid> key", () => {
    const key = newColumnKey();
    expect(key).toMatch(
      /^col_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
    expect(newColumnKey()).not.toBe(key);
  });
});
