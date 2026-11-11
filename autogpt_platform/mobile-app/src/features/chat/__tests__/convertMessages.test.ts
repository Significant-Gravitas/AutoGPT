import { toUIMessages } from "../convertMessages";

describe("toUIMessages", () => {
  it("turns a user and an assistant row into two messages", () => {
    const messages = toUIMessages("s1", [
      { role: "user", content: "Hi", sequence: 1 },
      { role: "assistant", content: "Hello", sequence: 2 },
    ]);

    expect(messages.map((m) => m.id)).toEqual(["s1-1", "s1-2"]);
    expect(messages.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(messages[0].parts).toEqual([
      { type: "text", text: "Hi", state: "done" },
    ]);
    expect(messages[1].parts).toEqual([
      { type: "text", text: "Hello", state: "done" },
    ]);
  });

  it("merges assistant and reasoning rows into one message in order", () => {
    const messages = toUIMessages("s1", [
      { role: "assistant", content: "First", sequence: 2 },
      { role: "reasoning", content: "Thinking", sequence: 3 },
      { role: "assistant", content: "Second", sequence: 4 },
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0].id).toBe("s1-4");
    expect(messages[0].role).toBe("assistant");
    expect(messages[0].parts.map((p) => p.type)).toEqual([
      "text",
      "reasoning",
      "text",
    ]);
  });

  it("pairs a tool call with its output", () => {
    const messages = toUIMessages("s1", [
      {
        role: "assistant",
        content: null,
        sequence: 2,
        tool_calls: [
          {
            id: "c1",
            display_name: "Search the web",
            function: { name: "web_search", arguments: '{"q":"x"}' },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "c1",
        content: '{"ok":true}',
        sequence: 3,
      },
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0].parts).toEqual([
      {
        type: "tool-web_search",
        toolCallId: "c1",
        title: "Search the web",
        input: { q: "x" },
        state: "output-available",
        output: { ok: true },
      },
    ]);
  });

  it("pairs a tool call in an older page with its output in the newer page", () => {
    const olderPage = [
      { role: "user", content: "Look it up", sequence: 1 },
      {
        role: "assistant",
        content: null,
        sequence: 2,
        tool_calls: [
          {
            id: "c1",
            function: { name: "web_search", arguments: '{"q":"x"}' },
          },
        ],
      },
    ];
    const newerPage = [
      { role: "tool", tool_call_id: "c1", content: '{"ok":true}', sequence: 3 },
      { role: "assistant", content: "Found it", sequence: 4 },
    ];

    const messages = toUIMessages("s1", [...olderPage, ...newerPage]);

    expect(messages.map((m) => m.id)).toEqual(["s1-1", "s1-4"]);
    expect(messages[1].parts[0]).toMatchObject({
      type: "tool-web_search",
      toolCallId: "c1",
      state: "output-available",
      output: { ok: true },
    });
    expect(messages[1].parts[1]).toEqual({
      type: "text",
      text: "Found it",
      state: "done",
    });
  });

  it("leaves out the rows of a running turn", () => {
    const messages = toUIMessages(
      "s1",
      [
        {
          role: "user",
          content: "Go",
          sequence: 1,
          created_at: "2026-11-10T09:59:59Z",
        },
        {
          role: "assistant",
          content: "Working",
          sequence: 2,
          created_at: "2026-11-10T10:00:01Z",
        },
        {
          role: "assistant",
          content: "Still working",
          sequence: 3,
          created_at: "2026-11-10T10:00:05Z",
        },
      ],
      { activeTurnStartedAt: "2026-11-10T10:00:00Z" },
    );

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe("user");
    expect(messages[0].id).toBe("s1-1");
  });

  it("drops the attached files block from a user message", () => {
    const messages = toUIMessages("s1", [
      {
        role: "user",
        content:
          "Read this\n\n[Attached files]\n- a.pdf (application/pdf, 12.0 KB), file_id=f1",
        sequence: 1,
      },
    ]);

    expect(messages[0].parts).toEqual([
      { type: "text", text: "Read this", state: "done" },
    ]);
  });

  it("ignores rows it cannot read and system rows", () => {
    const rows = [
      null,
      {},
      { role: "system", content: "You are Otto", sequence: 1 },
      { role: "user", content: "Hi", sequence: 2 },
    ];

    expect(() => toUIMessages("s1", rows)).not.toThrow();
    const messages = toUIMessages("s1", rows);
    expect(messages).toHaveLength(1);
    expect(messages[0].id).toBe("s1-2");
  });
});
