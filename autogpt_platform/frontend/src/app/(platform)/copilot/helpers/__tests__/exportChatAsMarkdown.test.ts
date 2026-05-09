import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { exportChatAsMarkdown } from "../exportChatAsMarkdown";

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
    expect(anchor.download).toContain("My-Chat");
  });

  it("falls back to 'Untitled-chat' when title is null", () => {
    exportChatAsMarkdown("session-1", null, []);
    expect(anchor.download).toContain("Untitled-chat");
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

  it("includes export date in the markdown body", () => {
    exportChatAsMarkdown("session-1", "Test", []);

    const blob: Blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    return blob.text().then((text) => {
      expect(text).toMatch(/_Exported: \d{4}-\d{2}-\d{2}_/);
    });
  });
});
