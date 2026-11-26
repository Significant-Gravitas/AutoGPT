import { chatActionOptions, validateTitle, withPinned, withRenamed, withoutSession } from "../chatActions";

const chat = (id: string, over = {}) => ({ id, title: id, is_pinned: false, updated_at: "2026-11-01T00:00:00Z", created_at: "2026-11-01T00:00:00Z", is_processing: false, expert_id: null, ...over });

test("offers Unpin for a pinned chat", () => {
  expect(chatActionOptions(chat("a", { is_pinned: true })).map((o) => o.label)).toEqual(["Rename", "Unpin", "Delete"]);
  expect(chatActionOptions(chat("a")).map((o) => o.label)).toEqual(["Rename", "Pin", "Delete"]);
});
test("refuses an empty or overlong name", () => {
  expect(validateTitle("   ")).toEqual({ error: "Give the chat a name." });
  expect(validateTitle("x".repeat(201))).toEqual({ error: "Keep the name under 200 characters." });
  expect(validateTitle("  Q4 plan ")).toEqual({ title: "Q4 plan" });
});
test("a pinned chat moves ahead of newer unpinned ones", () => {
  const list = { sessions: [chat("new", { updated_at: "2026-11-03T00:00:00Z" }), chat("old")], total: 2 };
  expect(withPinned(list, "old", true).sessions.map((s) => s.id)).toEqual(["old", "new"]);
});
test("rename and delete leave the other chats alone", () => {
  const list = { sessions: [chat("a"), chat("b")], total: 2 };
  expect(withRenamed(list, "a", "Hi").sessions[0].title).toBe("Hi");
  expect(withoutSession(list, "a")).toEqual({ sessions: [list.sessions[1]], total: 1 });
});
