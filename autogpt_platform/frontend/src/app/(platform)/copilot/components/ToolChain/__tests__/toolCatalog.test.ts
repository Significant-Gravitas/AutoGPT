import { describe, expect, it } from "vitest";
import { getCatalogLabel } from "../toolCatalog";
import { looksTechnical } from "../plainText";

describe("getCatalogLabel", () => {
  describe("bash_exec", () => {
    const purpose = "Updating your launch brief";
    const command = "cd ~/workspace/dewleaf && sed -i 's/TBD/25 Nov/' brief.md";

    it("labels the step with its plain-language purpose, never the command", () => {
      expect(
        getCatalogLabel("bash_exec", { purpose, command }, "running"),
      ).toEqual({ category: "bash", text: "Updating your launch brief…" });
      expect(
        getCatalogLabel("bash_exec", { purpose, command }, "done"),
      ).toEqual({
        category: "bash",
        text: "Updating your launch brief",
      });
      expect(
        getCatalogLabel("bash_exec", { purpose, command }, "error")?.text,
      ).toBe("Failed while updating your launch brief");
    });

    it("keeps the capital of a purpose that opens with a name", () => {
      expect(
        getCatalogLabel(
          "bash_exec",
          { purpose: "Shopify sync for the launch", command },
          "error",
        )?.text,
      ).toBe("Failed: Shopify sync for the launch");
    });

    it("falls back to the generic label when the purpose is itself technical", () => {
      for (const purpose of [
        "Patching brief.md with sed",
        "Running cd ~/workspace/dewleaf",
        "Calling run_capability",
      ]) {
        expect(
          getCatalogLabel("bash_exec", { purpose, command }, "done")?.text,
        ).toBe("Ran a command");
      }
    });

    it("never shows two ellipses on a running step", () => {
      const long = getCatalogLabel(
        "bash_exec",
        { purpose: "Checking ".repeat(20), command },
        "running",
      )?.text;
      expect(long?.endsWith("…")).toBe(true);
      expect(long).not.toContain("……");
      const trailing = { purpose: "Updating your brief…", command };
      expect(getCatalogLabel("bash_exec", trailing, "running")?.text).toBe(
        "Updating your brief…",
      );
      expect(getCatalogLabel("bash_exec", trailing, "done")?.text).toBe(
        "Updating your brief",
      );
    });

    it("tidies a purpose the model wrote as a sentence", () => {
      expect(
        getCatalogLabel(
          "bash_exec",
          { purpose: "  Updating   your launch brief.  ", command },
          "done",
        )?.text,
      ).toBe("Updating your launch brief");
    });

    it("caps a long purpose", () => {
      const text =
        getCatalogLabel(
          "bash_exec",
          { purpose: "Checking ".repeat(20), command },
          "done",
        )?.text ?? "";
      expect(text.length).toBeLessThanOrEqual(80);
      expect(text.endsWith("…")).toBe(true);
    });

    it("falls back to a generic label for calls saved without a purpose", () => {
      for (const input of [
        { command },
        { command, purpose: "  " },
        undefined,
      ]) {
        expect(getCatalogLabel("bash_exec", input, "running")?.text).toBe(
          "Running a command…",
        );
        expect(getCatalogLabel("bash_exec", input, "done")?.text).toBe(
          "Ran a command",
        );
      }
      expect(getCatalogLabel("bash_exec", { command }, "error")?.text).toBe(
        "Failed while running a command",
      );
    });
  });

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
    [
      "browser_navigate",
      { url: "https://agpt.co" },
      "running",
      'Opening "https://agpt.co"…',
    ],
    ["read_workspace_file", { path: "notes.md" }, "done", 'Read "Notes"'],
    ["read_workspace_file", { filename: "notes.md" }, "done", 'Read "Notes"'],
    [
      "write_workspace_file",
      { filename: "out.csv" },
      "running",
      'Writing "Out"…',
    ],
    ["delete_workspace_file", { path: "tmp.txt" }, "done", 'Deleted "Tmp"'],
    ["delete_workspace_file", { filename: "tmp.txt" }, "done", 'Deleted "Tmp"'],
    [
      "post_to_chat_platform",
      { channel: "#general" },
      "done",
      'Posted to "#general"',
    ],
    ["run_block", { block_name: "Web Search" }, "done", "Ran block"],
    [
      "run_block",
      { block_id: "abcdefabcdefabcdefabcdef" },
      "done",
      "Ran block",
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
      'Finding workflows for "scraper"…',
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
      "Ran workflow",
    ],
    [
      "run_agent",
      { library_agent_id: "0123456789012345678901234" },
      "done",
      "Ran workflow",
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

    expect(text.startsWith('Delegating subtask: "Investigate')).toBe(true);
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

  it("labels the expert team tools", () => {
    expect(
      getCatalogLabel("hire_expert", { template_id: "tpl" }, "done"),
    ).toEqual({ category: "team", text: "Ready to hire" });
    expect(getCatalogLabel("raise_expert", { name: "Otto" }, "done")).toEqual({
      category: "team",
      text: 'Ready to create "Otto"',
    });
    expect(
      getCatalogLabel(
        "confirm_expert_change",
        { confirmation_id: "c" },
        "done",
      ),
    ).toEqual({ category: "team", text: "Added to the team" });
    expect(
      getCatalogLabel("handoff_to_expert", { prompt: "own it" }, "done")?.text,
    ).toBe('Handed over: "own it"');
    expect(
      getCatalogLabel(
        "consult_teammate",
        { question: "Does this commit us to a refund?" },
        "running",
      ),
    ).toEqual({
      category: "team",
      text: 'Asking a teammate to check: "Does this commit us to a refund?"…',
    });
    expect(
      getCatalogLabel(
        "consult_teammate",
        { question: "Does this commit us to a refund?" },
        "done",
      )?.text,
    ).toBe('Teammate checked: "Does this commit us to a refund?"');
    expect(getCatalogLabel("find_session", {}, "running")).toEqual({
      category: "team",
      text: "Looking for a session…",
    });
    expect(getCatalogLabel("find_session", {}, "done")?.text).toBe(
      "Found sessions",
    );
    // The message is quoted and truncated, so the row never carries the whole
    // payload into the chain label.
    expect(
      getCatalogLabel(
        "message_session",
        { message: "the numbers are in" },
        "done",
      )?.text,
    ).toBe('Messaged a session: "the numbers are in"');
  });

  it("labels the capability tools by what they act on", () => {
    expect(
      getCatalogLabel("find_capability", { query: "linear issue" }, "running")
        ?.text,
    ).toBe('Searching capabilities for "linear issue"…');
    expect(
      getCatalogLabel("describe_capability", { id: "block:abc" }, "done")?.text,
    ).toBe("Looked up a capability");
    expect(
      getCatalogLabel("describe_capability", { id: "block:abc" }, "done", {
        displayName: "Send Web Request",
      })?.text,
    ).toBe('Looked up "Send Web Request"');
  });

  it("names a run by its block, never by the capability id", () => {
    // The display name only arrives once the run reports back; until then
    // the row says what kind of step it is.
    expect(
      getCatalogLabel(
        "run_capability",
        { id: "tool:list_schedules" },
        "running",
      )?.text,
    ).toBe("Running a capability…");
    expect(
      getCatalogLabel("run_capability", { id: "block:abc" }, "error")?.text,
    ).toBe("Failed while running a capability");
    expect(
      getCatalogLabel("run_capability", { id: "block:abc" }, "done", {
        displayName: "Send Web Request",
      })?.text,
    ).toBe('Ran "Send Web Request"');
    expect(
      getCatalogLabel("resume_capability", { review_id: "r" }, "done", {
        displayName: "Send Web Request",
      })?.text,
    ).toBe('Resumed "Send Web Request"');
  });
});

describe("file and capability rows in plain language", () => {
  it("names a file by its humanized name, never its path", () => {
    const path = "~/workspace/dewleaf/launch_brief.md";
    expect(getCatalogLabel("read_workspace_file", { path }, "done")?.text).toBe(
      'Read "Launch Brief"',
    );
    expect(
      getCatalogLabel("write_workspace_file", { path }, "error")?.text,
    ).toBe('Failed while writing "Launch Brief"');
    expect(
      getCatalogLabel(
        "write_workspace_file",
        { filename: "Q3-budget.xlsx" },
        "done",
      )?.text,
    ).toBe('Wrote "Q3 Budget"');
  });

  it("says 'a file' when there is no name to show", () => {
    expect(getCatalogLabel("read_workspace_file", {}, "done")?.text).toBe(
      "Read a file",
    );
    expect(
      getCatalogLabel("delete_workspace_file", { path: "/tmp/" }, "running")
        ?.text,
    ).toBe("Deleting a file…");
  });

  it.each([
    ["read_workspace_file", { path: "/home/user/src/app/page.tsx" }],
    ["write_workspace_file", { filename: "~/workspace/out/data.json" }],
    ["delete_workspace_file", { path: "./build/index.js" }],
    ["describe_capability", { id: "block:6f1c2d3e-aaaa-bbbb-cccc-1234" }],
    ["run_capability", { id: "tool:list_schedules" }],
  ])("shows no path or id on a %s row", (tool, input) => {
    for (const state of ["running", "done", "error"] as const) {
      const text = getCatalogLabel(tool, input, state)?.text ?? "";
      expect(looksTechnical(text)).toBe(false);
      expect(text).not.toMatch(/[/:]|\.(?:md|json|js|tsx)\b/);
    }
  });
});
