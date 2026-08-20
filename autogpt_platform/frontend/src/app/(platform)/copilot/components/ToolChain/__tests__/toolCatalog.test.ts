import { describe, expect, it } from "vitest";
import { getCatalogLabel } from "../toolCatalog";

describe("getCatalogLabel", () => {
  it("returns null for tools missing from the catalog", () => {
    expect(getCatalogLabel("totally_unknown_tool", {}, "running")).toBeNull();
  });

  it("builds running, done and error labels around the subject", () => {
    const input = { query: "copilot news" };

    expect(getCatalogLabel("web_search", input, "running")).toEqual({
      category: "web",
      text: 'Searching the web for "copilot news"…',
    });
    expect(getCatalogLabel("web_search", input, "done")).toEqual({
      category: "web",
      text: 'Searched the web for "copilot news"',
    });
    expect(getCatalogLabel("web_search", input, "error")).toEqual({
      category: "web",
      text: 'Failed while searching the web for "copilot news"',
    });
  });

  it("omits the subject when input is not an object", () => {
    expect(getCatalogLabel("web_search", undefined, "running")?.text).toBe(
      "Searching the web for…",
    );
    expect(getCatalogLabel("web_search", "raw", "done")?.text).toBe(
      "Searched the web for",
    );
  });

  it("truncates long subjects with an ellipsis", () => {
    const url = `https://example.com/${"a".repeat(80)}`;
    const text = getCatalogLabel("web_fetch", { url }, "done")?.text ?? "";

    expect(text.startsWith('Fetched "https://example.com/')).toBe(true);
    expect(text.endsWith('…"')).toBe(true);
    expect(text.length).toBeLessThan(url.length);
  });

  it.each([
    ["bash_exec", { command: "ls -la" }, "done", 'Ran command "ls -la"'],
    [
      "browser_navigate",
      { url: "https://agpt.co" },
      "running",
      'Opening "https://agpt.co"…',
    ],
    ["read_workspace_file", { path: "notes.md" }, "done", 'Read "notes.md"'],
    [
      "read_workspace_file",
      { filename: "notes.md" },
      "done",
      'Read "notes.md"',
    ],
    [
      "write_workspace_file",
      { filename: "out.csv" },
      "running",
      'Writing "out.csv"…',
    ],
    ["delete_workspace_file", { path: "tmp.txt" }, "done", 'Deleted "tmp.txt"'],
    [
      "delete_workspace_file",
      { filename: "tmp.txt" },
      "done",
      'Deleted "tmp.txt"',
    ],
    [
      "post_to_chat_platform",
      { channel: "#general" },
      "done",
      'Posted to "#general"',
    ],
    [
      "run_block",
      { block_name: "Web Search" },
      "done",
      'Ran block "Web Search"',
    ],
    [
      "run_block",
      { block_id: "abcdefabcdefabcdefabcdef" },
      "done",
      'Ran block "abcdefabcdefabcdefab…"',
    ],
    [
      "run_mcp_tool",
      { tool_name: "list_issues" },
      "done",
      'Ran MCP tool "list_issues"',
    ],
    [
      "search_docs",
      { query: "webhooks" },
      "done",
      'Searched docs for "webhooks"',
    ],
    [
      "get_doc_page",
      { path: "platform/blocks" },
      "done",
      "Read doc page platform/blocks",
    ],
    ["store_skill", { name: "scraper" }, "done", 'Saved skill "scraper"'],
    ["read_skill", { name: "scraper" }, "running", 'Reading skill "scraper"…'],
    ["delete_skill", { name: "scraper" }, "done", 'Deleted skill "scraper"'],
    [
      "connect_integration",
      { provider: "github" },
      "running",
      "Connecting github…",
    ],
    [
      "search_feature_requests",
      { query: "dark mode" },
      "done",
      'Searched feature requests for "dark mode"',
    ],
    [
      "create_feature_request",
      { title: "Dark mode" },
      "done",
      'Filed feature request "Dark mode"',
    ],
    ["TodoWrite", {}, "done", "Updated tasks"],
    ["TodoWrite", {}, "error", "Failed while updating tasks"],
    ["get_platform_info", {}, "running", "Checking platform info…"],
  ])("labels platform tool %s", (tool, input, state, expected) => {
    expect(
      getCatalogLabel(tool, input, state as "running" | "done" | "error")?.text,
    ).toBe(expected);
  });

  it.each([
    [
      "decompose_goal",
      { goal: "Track prices" },
      "done",
      'Broke down the goal "Track prices"',
    ],
    ["find_block", { query: "email" }, "done", 'Found blocks for "email"'],
    [
      "find_agent",
      { query: "scraper" },
      "running",
      'Finding agents for "scraper"…',
    ],
    [
      "find_library_agent",
      { query: "newsletter" },
      "done",
      'Searched your library for "newsletter"',
    ],
    [
      "memory_search",
      { query: "budget" },
      "done",
      'Searched memory for "budget"',
    ],
    [
      "memory_store",
      { name: "preferences" },
      "done",
      'Stored memory "preferences"',
    ],
    ["create_folder", { name: "Reports" }, "done", 'Created folder "Reports"'],
    ["update_folder", { name: "Reports" }, "done", 'Updated folder "Reports"'],
    [
      "run_agent",
      { username_agent_slug: "abhi/scraper" },
      "done",
      'Ran agent "abhi/scraper"',
    ],
    [
      "run_agent",
      { library_agent_id: "0123456789012345678901234" },
      "done",
      'Ran agent "01234567890123456789…"',
    ],
    ["add_understanding", {}, "done", "Noted context"],
    ["enter_agent_building_mode", {}, "running", "Entering building mode…"],
  ])("labels agent tool %s", (tool, input, state, expected) => {
    expect(
      getCatalogLabel(tool, input, state as "running" | "done" | "error")?.text,
    ).toBe(expected);
  });

  it("truncates run_sub_session prompts to 45 characters", () => {
    const prompt = "Investigate every failing pipeline in the workspace today";
    const text =
      getCatalogLabel("run_sub_session", { prompt }, "running")?.text ?? "";

    expect(text.startsWith('Delegating to sub-AutoPilot: "Investigate')).toBe(
      true,
    );
    expect(text.endsWith('…"…')).toBe(true);
  });

  it("reads ask_question subjects from the direct field or the array", () => {
    expect(
      getCatalogLabel("ask_question", { question: "Which region?" }, "done")
        ?.text,
    ).toBe('Asked you a question "Which region?"');

    expect(
      getCatalogLabel(
        "ask_question",
        { questions: [{ question: "Which format?" }] },
        "done",
      )?.text,
    ).toBe('Asked you a question "Which format?"');

    expect(
      getCatalogLabel("ask_question", { questions: ["plain"] }, "done")?.text,
    ).toBe("Asked you a question");

    expect(getCatalogLabel("ask_question", {}, "done")?.text).toBe(
      "Asked you a question",
    );
  });

  it("ignores blank and non-string subject values", () => {
    expect(getCatalogLabel("web_search", { query: "   " }, "done")?.text).toBe(
      "Searched the web for",
    );
    expect(getCatalogLabel("web_search", { query: 42 }, "done")?.text).toBe(
      "Searched the web for",
    );
  });
});
