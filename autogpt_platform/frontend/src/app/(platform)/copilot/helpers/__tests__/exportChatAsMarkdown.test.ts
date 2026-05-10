import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  exportChatAsMarkdown,
  fetchAndExportChat,
} from "../exportChatAsMarkdown";

describe("exportChatAsMarkdown", () => {
  let clickSpy: ReturnType<typeof vi.fn>;
  let removeSpy: ReturnType<typeof vi.fn>;
  let anchor: {
    href: string;
    download: string;
    click: ReturnType<typeof vi.fn>;
    remove: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    clickSpy = vi.fn();
    removeSpy = vi.fn();

    anchor = { href: "", download: "", click: clickSpy, remove: removeSpy };

    vi.stubGlobal(
      "URL",
      Object.assign(URL, {
        createObjectURL: vi.fn().mockReturnValue("blob:fake-url"),
        revokeObjectURL: vi.fn(),
      }),
    );

    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    vi.spyOn(document.body, "appendChild").mockImplementation(
      (node) => node as ChildNode,
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("triggers a download with .md extension", () => {
    exportChatAsMarkdown("session-1", "My Chat", []);
    expect(clickSpy).toHaveBeenCalled();
    expect(removeSpy).toHaveBeenCalled();
    expect(anchor.download).toMatch(/\.md$/);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:fake-url");
  });

  it("includes the chat title in the filename", () => {
    exportChatAsMarkdown("session-1", "My Chat", []);
    expect(anchor.download).toContain("My Chat");
  });

  it("falls back to 'Untitled chat' when title is null", () => {
    exportChatAsMarkdown("session-1", null, []);
    expect(anchor.download).toContain("Untitled chat");
  });

  it("renders user and assistant messages with headers", () => {
    const messages = [
      { role: "user", content: "Hello", tool_calls: null },
      { role: "assistant", content: "Hi there", tool_calls: null },
    ];
    exportChatAsMarkdown("session-1", "Test", messages);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).toContain("## User");
      expect(text).toContain("Hello");
      expect(text).toContain("## Assistant");
      expect(text).toContain("Hi there");
    });
  });

  it("skips tool result messages (role === 'tool')", () => {
    const messages = [
      { role: "tool", content: "tool result", tool_calls: null },
    ];
    exportChatAsMarkdown("session-1", "Test", messages);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).not.toContain("tool result");
      expect(text).not.toContain("## Tool");
    });
  });

  it("renders tool calls as blockquotes with emoji", () => {
    const messages = [
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            function: {
              name: "search_web",
              arguments: JSON.stringify({ query: "cats" }),
            },
          },
        ],
      },
    ];
    exportChatAsMarkdown("session-1", "Test", messages);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).toContain("> 🔧");
      expect(text).toContain("search_web");
    });
  });

  it("sanitizes path traversal in title", () => {
    exportChatAsMarkdown("session-1", "../../etc/passwd", []);
    expect(anchor.download).not.toContain("..");
    expect(anchor.download).not.toContain("/");
  });

  it("falls back to 'Untitled chat' when title is undefined", () => {
    exportChatAsMarkdown("session-1", undefined, []);
    expect(anchor.download).toContain("Untitled chat");
  });

  it("uses 'unknown_tool' when tool call has no function name", () => {
    const messages = [
      {
        role: "assistant",
        content: null,
        tool_calls: [{ function: { name: undefined, arguments: "{}" } }],
      },
    ];
    exportChatAsMarkdown("session-1", "Test", messages);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).toContain("unknown_tool");
    });
  });

  it("handles invalid JSON in tool call arguments gracefully", () => {
    const messages = [
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { function: { name: "my_tool", arguments: "not-valid-json" } },
        ],
      },
    ];
    exportChatAsMarkdown("session-1", "Test", messages);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).toContain("my_tool");
      expect(text).toContain("not-valid-json");
    });
  });

  it("includes export date in the markdown body", () => {
    exportChatAsMarkdown("session-1", "Test", []);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).toMatch(/_Exported: \d{4}-\d{2}-\d{2}_/);
    });
  });
});

describe("fetchAndExportChat", () => {
  let clickSpy: ReturnType<typeof vi.fn>;
  let anchor: {
    href: string;
    download: string;
    click: ReturnType<typeof vi.fn>;
    remove: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    clickSpy = vi.fn();
    anchor = { href: "", download: "", click: clickSpy, remove: vi.fn() };

    vi.stubGlobal(
      "URL",
      Object.assign(URL, {
        createObjectURL: vi.fn().mockReturnValue("blob:fake-url"),
        revokeObjectURL: vi.fn(),
      }),
    );

    vi.spyOn(document, "createElement").mockReturnValue(
      anchor as unknown as HTMLAnchorElement,
    );

    vi.spyOn(document.body, "appendChild").mockImplementation(
      (node) => node as ChildNode,
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("calls exportChatAsMarkdown when fetch succeeds", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      status: 200,
      data: {
        messages: [{ role: "user", content: "Hello", tool_calls: null }],
      },
    });
    await fetchAndExportChat("session-1", "My Chat", mockFetch);
    expect(clickSpy).toHaveBeenCalled();
    expect(anchor.download).toMatch(/\.md$/);
  });

  it("throws when fetch returns non-200 status (includes status code)", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ status: 404, data: {} });
    await expect(
      fetchAndExportChat("session-1", "My Chat", mockFetch),
    ).rejects.toThrow("Failed to fetch session (status: 404)");
  });

  it("handles null messages array gracefully", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      status: 200,
      data: { messages: null },
    });
    await fetchAndExportChat("session-1", "My Chat", mockFetch);
    expect(clickSpy).toHaveBeenCalled();
  });

  it("paginates via before_sequence until has_more_messages is false", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce({
        status: 200,
        data: {
          messages: [
            { role: "user", content: "page2-msg-a", tool_calls: null },
            { role: "assistant", content: "page2-msg-b", tool_calls: null },
          ],
          has_more_messages: true,
          oldest_sequence: 42,
        },
      })
      .mockResolvedValueOnce({
        status: 200,
        data: {
          messages: [
            { role: "user", content: "page1-msg-a", tool_calls: null },
            { role: "assistant", content: "page1-msg-b", tool_calls: null },
          ],
          has_more_messages: false,
          oldest_sequence: 1,
        },
      });

    await fetchAndExportChat("session-1", "My Chat", mockFetch);

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch).toHaveBeenNthCalledWith(1, "session-1", { limit: 200 });
    expect(mockFetch).toHaveBeenNthCalledWith(2, "session-1", {
      limit: 200,
      before_sequence: 42,
    });
    expect(clickSpy).toHaveBeenCalled();
  });

  it("throws when pagination cap is hit while has_more is still true", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      status: 200,
      data: {
        messages: [{ role: "user", content: "msg", tool_calls: null }],
        has_more_messages: true,
        oldest_sequence: 1,
      },
    });

    await expect(
      fetchAndExportChat("session-1", "My Chat", mockFetch),
    ).rejects.toThrow(/exceeded \d+ messages/);

    expect(clickSpy).not.toHaveBeenCalled();
  });

  it("stops paginating when oldest_sequence is null even if has_more is true", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      status: 200,
      data: {
        messages: [{ role: "user", content: "Hello", tool_calls: null }],
        has_more_messages: true,
        oldest_sequence: null,
      },
    });

    await fetchAndExportChat("session-1", "My Chat", mockFetch);

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(clickSpy).toHaveBeenCalled();
  });
});
