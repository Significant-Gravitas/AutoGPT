import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";
import type { ChainRow } from "../helpers";
import { ToolResult } from "../ToolResult";

vi.mock("../../SetupRequirementsCard/SetupRequirementsCard", () => ({
  SetupRequirementsCard: ({
    credentialsLabel,
    inputsMode,
    retryInstruction,
  }: {
    credentialsLabel?: string;
    inputsMode?: string;
    retryInstruction?: string;
  }) => (
    <div>
      <span>{credentialsLabel}</span>
      {inputsMode && <span>{`mode:${inputsMode}`}</span>}
      {retryInstruction && <span>{`retry:${retryInstruction}`}</span>}
    </div>
  ),
}));

vi.mock(
  "../../../tools/RunMCPTool/components/MCPSetupCard/MCPSetupCard",
  () => ({
    MCPSetupCard: () => <div>mcp-setup-card</div>,
  }),
);

vi.mock("../../QuestionDock/QuestionDock", () => ({
  QuestionsForm: ({ dockId }: { dockId: string }) => (
    <div>{`questions-form:${dockId}`}</div>
  ),
}));

function row(output: unknown, tool?: string, input?: unknown): ChainRow {
  return {
    key: "tool",
    category: "other",
    text: "Tool",
    state: "done",
    output,
    tool,
    input,
  };
}

describe("ToolResult", () => {
  afterEach(cleanup);

  it("renders a real unified diff", () => {
    render(<ToolResult row={row("@@ -1 +1 @@\n-old\n+new")} />);

    expect(screen.getByText("+1")).toBeDefined();
    expect(screen.getByText("-1")).toBeDefined();
    expect(screen.getByText("old")).toBeDefined();
    expect(screen.getByText("new")).toBeDefined();
  });

  it("renders JSON containing diff-like text as structured output", () => {
    render(
      <ToolResult
        row={row(JSON.stringify({ patch: "@@ -1 +1 @@\n-old\n+new" }))}
      />,
    );

    expect(screen.getByText("Patch")).toBeDefined();
    expect(screen.queryByText("+1")).toBeNull();
  });

  it("uses a fallback credentials label when an integration name is missing", () => {
    render(
      <ToolResult
        row={row(
          {
            message: "Connect an account",
            setup_info: { agent_id: "agent-1" },
          },
          "connect_integration",
        )}
      />,
    );

    expect(screen.getByText("Integration credentials")).toBeDefined();
    expect(screen.queryByText("undefined credentials")).toBeNull();
  });

  it.each([
    [
      "run_agent",
      { graph_name: "Research Agent", execution_id: "exec-1" },
      "Research Agent",
    ],
    ["find_block", { blocks: [{ name: "Web Search" }] }, "Web Search"],
    [
      "web_search",
      { results: [{ title: "Example result", url: "https://example.com" }] },
      "Example result",
    ],
    [
      "ask_question",
      { questions: [{ question: "Which region?", keyword: "region" }] },
      "Which region?",
    ],
    [
      "search_feature_requests",
      { results: [{ identifier: "FR-101", title: "Dark mode" }] },
      "Dark mode",
    ],
    [
      "create_agent",
      { agent_name: "Draft agent", node_count: 3 },
      "Draft agent",
    ],
    ["run_mcp_tool", { result: { identifier: "OPEN-3211" } }, "OPEN-3211"],
  ])("dispatches %s to its specialized card", (tool, output, expected) => {
    render(<ToolResult row={row(output, tool)} />);

    expect(screen.getByText(expected)).toBeDefined();
  });

  describe("setup requirements routing", () => {
    it("routes run_mcp_tool setup to the MCP setup card", () => {
      render(
        <ToolResult
          row={row({ setup_info: { server_url: "url" } }, "run_mcp_tool")}
        />,
      );

      expect(screen.getByText("mcp-setup-card")).toBeDefined();
    });

    it("routes trigger setup with trigger inputs mode and Account label", () => {
      render(
        <ToolResult
          row={row(
            { setup_info: { agent_name: "Slack" } },
            "setup_agent_webhook_trigger",
          )}
        />,
      );

      expect(screen.getByText("Account")).toBeDefined();
      expect(screen.getByText("mode:trigger")).toBeDefined();
    });

    it("routes run_agent setup with preview inputs mode", () => {
      render(
        <ToolResult
          row={row({ setup_info: { agent_name: "Runner" } }, "run_agent")}
        />,
      );

      expect(screen.getByText("mode:preview")).toBeDefined();
    });

    it("labels connect_integration setup with the integration name and retry note", () => {
      render(
        <ToolResult
          row={row(
            { setup_info: { agent_name: "Notion" } },
            "connect_integration",
          )}
        />,
      );

      expect(screen.getByText("Notion credentials")).toBeDefined();
      expect(
        screen.getByText("retry:I've connected my account. Please continue."),
      ).toBeDefined();
    });
  });

  describe("pending clarifying questions", () => {
    it("renders the interactive answer form for the pending row", () => {
      render(
        <PendingQuestionsContext.Provider
          value={{
            dockId: "dock-1",
            questions: [{ question: "Which region?", keyword: "region" }],
            callIds: ["tool"],
          }}
        >
          <ToolResult
            row={row({ questions: [{ question: "Which region?" }] })}
          />
        </PendingQuestionsContext.Provider>,
      );

      expect(screen.getByText("questions-form:dock-1")).toBeDefined();
    });

    it("keeps a read-only card for ask_question rows answered from input", () => {
      render(
        <ToolResult
          row={row({ type: "clarification" }, "ask_question", {
            questions: [
              { question: "What budget?", keyword: "budget", example: "$50" },
            ],
          })}
        />,
      );

      expect(screen.getByText("What budget?")).toBeDefined();
      expect(screen.getByText("e.g. $50")).toBeDefined();
    });
  });

  describe("agent execution routing", () => {
    it("renders an execution card with the library agent link", () => {
      render(
        <ToolResult
          row={row(
            {
              graph_name: "Research Agent",
              execution_id: "exec-1",
              status: "RUNNING",
              library_agent_link: "/library/agents/lib-1",
            },
            "run_agent",
          )}
        />,
      );

      expect(screen.getByText("Research Agent")).toBeDefined();
      expect(screen.getByText("running")).toBeDefined();
      expect(screen.getByLabelText("Open execution").getAttribute("href")).toBe(
        "/library/agents/lib-1",
      );
    });

    it("builds the execution link from graph and execution ids", () => {
      render(
        <ToolResult
          row={row(
            { execution_id: "exec-2", graph_id: "graph-9" },
            "schedule_agent",
            { username_agent_slug: "creator/scraper" },
          )}
        />,
      );

      expect(screen.getByText("creator/scraper")).toBeDefined();
      expect(screen.getByLabelText("Open execution").getAttribute("href")).toBe(
        "/library/agents/graph-9?activeTab=runs&activeItem=exec-2",
      );
    });

    it("renders the agent card when run_agent returns an agent object", () => {
      render(
        <ToolResult
          row={row(
            { agent: { name: "Found Agent", description: "Scrapes" } },
            "run_agent",
          )}
        />,
      );

      expect(screen.getByText("Found Agent")).toBeDefined();
      expect(screen.getByText("Scrapes")).toBeDefined();
    });

    it("renders execution outputs when run_agent finishes with outputs", () => {
      render(
        <ToolResult
          row={row(
            { execution: { outputs: { summary: ["All good"] } } },
            "run_agent",
          )}
        />,
      );

      expect(screen.getByText("summary")).toBeDefined();
      expect(screen.getByText("All good")).toBeDefined();
    });

    it("renders view_agent_output execution outputs", () => {
      render(
        <ToolResult
          row={row(
            { execution: { outputs: { answer: [42] } } },
            "view_agent_output",
          )}
        />,
      );

      expect(screen.getByText("answer")).toBeDefined();
      expect(screen.getByText("42")).toBeDefined();
    });

    it("renders view_agent_output top-level outputs list", () => {
      render(
        <ToolResult
          row={row(
            { outputs: [{ name: "report", value: "Done" }] },
            "view_agent_output",
          )}
        />,
      );

      expect(screen.getByText("report")).toBeDefined();
      expect(screen.getByText("Done")).toBeDefined();
    });
  });

  describe("agent editing routing", () => {
    it("renders the suggested goal card", () => {
      render(
        <ToolResult
          row={row(
            { suggested_goal: "Build a scraper", reason: "You asked for it" },
            "create_agent",
          )}
        />,
      );

      expect(screen.getByText("Build a scraper")).toBeDefined();
      expect(screen.getByText("You asked for it")).toBeDefined();
    });

    it("renders the saved agent card with builder and library links", () => {
      render(
        <ToolResult
          row={row(
            {
              agent_name: "My Agent",
              graph_version: 2,
              library_agent_link: "/library/agents/lib-2",
              agent_page_link: "/build?flowID=graph-2",
            },
            "edit_agent",
          )}
        />,
      );

      expect(screen.getByText("My Agent")).toBeDefined();
      expect(screen.getByText("v2")).toBeDefined();
      expect(
        screen.getByLabelText("Open in builder").getAttribute("href"),
      ).toBe("/build?flowID=graph-2");
      expect(
        screen.getByLabelText("Open in library").getAttribute("href"),
      ).toBe("/library/agents/lib-2");
    });

    it("renders the sub-session card with status and elapsed time", () => {
      render(
        <ToolResult
          row={row(
            {
              status: "COMPLETED",
              response: "Everything worked",
              elapsed_seconds: 75,
              sub_autopilot_session_link: "/copilot?session=sub-1",
            },
            "run_sub_session",
          )}
        />,
      );

      expect(screen.getByText("Sub-AutoPilot")).toBeDefined();
      expect(screen.getByText("1m 15s")).toBeDefined();
      expect(screen.getByText("Everything worked")).toBeDefined();
      expect(
        screen.getByLabelText("Open sub-session").getAttribute("href"),
      ).toBe("/copilot?session=sub-1");
    });

    it("renders found library agents with stats", () => {
      render(
        <ToolResult
          row={row(
            {
              agents: [
                {
                  id: "lib-1",
                  source: "library",
                  name: "Lib Agent",
                  creator: "abhi",
                  runs: 1200,
                  rating: 4.5,
                },
              ],
            },
            "find_library_agent",
          )}
        />,
      );

      expect(screen.getByText("Lib Agent")).toBeDefined();
      expect(screen.getByText("by abhi")).toBeDefined();
      expect(screen.getByText("1,200 runs")).toBeDefined();
      expect(screen.getByText("4.5")).toBeDefined();
      expect(screen.getByLabelText("Open agent").getAttribute("href")).toBe(
        "/library/agents/lib-1",
      );
    });
  });

  describe("block routing", () => {
    it("renders a single block card for run_block block payloads", () => {
      render(
        <ToolResult
          row={row(
            {
              block: {
                name: "HTTP Request",
                description: "Makes requests",
                categories: ["NETWORK"],
              },
            },
            "run_block",
          )}
        />,
      );

      expect(screen.getByText("HTTP Request")).toBeDefined();
      expect(screen.getByText("Makes requests")).toBeDefined();
      expect(screen.getByText("network")).toBeDefined();
    });

    it("renders block outputs for continue_run_block results", () => {
      render(
        <ToolResult
          row={row(
            {
              block_name: "Send Email",
              success: false,
              outputs: { error_message: ["SMTP down"] },
            },
            "continue_run_block",
          )}
        />,
      );

      expect(screen.getByText("Send Email")).toBeDefined();
      expect(screen.getByText("error message")).toBeDefined();
      expect(screen.getByText("SMTP down")).toBeDefined();
    });
  });

  describe("plan, validation and skills routing", () => {
    it("renders decomposed goal steps", () => {
      render(
        <ToolResult
          row={row(
            {
              steps: [
                {
                  description: "Fetch data",
                  status: "completed",
                  block_name: "HTTP",
                },
                { description: "Summarize", status: "in_progress" },
              ],
            },
            "decompose_goal",
          )}
        />,
      );

      expect(screen.getByText("Fetch data")).toBeDefined();
      expect(screen.getByText("Summarize")).toBeDefined();
      expect(screen.getByText("HTTP")).toBeDefined();
    });

    it("renders a valid graph status for validate_agent_graph", () => {
      render(<ToolResult row={row({ valid: true }, "validate_agent_graph")} />);

      expect(screen.getByText("Graph is valid")).toBeDefined();
    });

    it("renders fix results for fix_agent_graph", () => {
      render(
        <ToolResult
          row={row(
            { valid_after_fix: true, fixes_applied: ["Linked input"] },
            "fix_agent_graph",
          )}
        />,
      );

      expect(screen.getByText("Fixed — applied 1 fix")).toBeDefined();
    });

    it("renders a stored skill card", () => {
      render(
        <ToolResult
          row={row(
            {
              name: "Weekly digest",
              description: "Summarizes the week",
              triggers: ["every friday"],
            },
            "store_skill",
          )}
        />,
      );

      expect(screen.getByText("Weekly digest")).toBeDefined();
      expect(screen.getByText("Summarizes the week")).toBeDefined();
      expect(screen.getByText("every friday")).toBeDefined();
    });

    it("renders skill names as chips for list_skills", () => {
      render(
        <ToolResult
          row={row(
            { skills: [{ name: "summarize" }, { name: "draft" }] },
            "list_skills",
          )}
        />,
      );

      expect(screen.getByText("Skills")).toBeDefined();
      expect(screen.getByText("summarize")).toBeDefined();
      expect(screen.getByText("draft")).toBeDefined();
    });

    it("renders channel names with a hash prefix", () => {
      render(
        <ToolResult
          row={row(
            { channels: [{ name: "general" }] },
            "list_chat_platform_channels",
          )}
        />,
      );

      expect(screen.getByText("Channels")).toBeDefined();
      expect(screen.getByText("#general")).toBeDefined();
    });
  });

  describe("schedules, folders, files and docs routing", () => {
    it("renders the schedule list with a chat kind chip", () => {
      render(
        <ToolResult
          row={row(
            {
              schedules: [
                {
                  name: "Daily run",
                  next_run_time: "2026-08-21T10:00:00Z",
                  cron: "0 10 * * *",
                  kind: "copilot_turn",
                },
              ],
            },
            "list_schedules",
          )}
        />,
      );

      expect(screen.getByText("Daily run")).toBeDefined();
      expect(screen.getByText("chat")).toBeDefined();
    });

    it("renders the follow-up scheduled card", () => {
      render(
        <ToolResult
          row={row(
            { next_run_time: "2026-08-21T10:00:00Z", is_recurring: true },
            "schedule_followup",
          )}
        />,
      );

      expect(screen.getByText("Follow-up scheduled")).toBeDefined();
    });

    it("renders folders for list_folders", () => {
      render(
        <ToolResult
          row={row(
            { folders: [{ name: "Marketing", agent_count: 3 }] },
            "list_folders",
          )}
        />,
      );

      expect(screen.getByText("Marketing")).toBeDefined();
      expect(screen.getByText("3 agents")).toBeDefined();
    });

    it("renders the single folder returned by create_folder", () => {
      render(
        <ToolResult
          row={row({ folder: { name: "New Folder" } }, "create_folder")}
        />,
      );

      expect(screen.getByText("New Folder")).toBeDefined();
    });

    it("renders workspace files with sizes", () => {
      render(
        <ToolResult
          row={row(
            {
              files: [
                { path: "chart.png", mime_type: "image/png", size_bytes: 2048 },
              ],
            },
            "list_workspace_files",
          )}
        />,
      );

      expect(screen.getByText("chart.png")).toBeDefined();
      expect(screen.getByText("2.0 KB")).toBeDefined();
    });

    it("renders docs search results with a link", () => {
      render(
        <ToolResult
          row={row(
            {
              results: [
                {
                  title: "Blocks",
                  section: "Guide",
                  snippet: "How blocks work",
                  doc_url: "https://docs.agpt.co/blocks",
                },
              ],
            },
            "search_docs",
          )}
        />,
      );

      expect(screen.getByText("Blocks")).toBeDefined();
      expect(screen.getByText("Guide")).toBeDefined();
      expect(screen.getByText("How blocks work")).toBeDefined();
      expect(screen.getByLabelText("Open doc").getAttribute("href")).toBe(
        "https://docs.agpt.co/blocks",
      );
    });

    it("renders a single doc page as a docs card", () => {
      render(
        <ToolResult row={row({ title: "Getting Started" }, "get_doc_page")} />,
      );

      expect(screen.getByText("Getting Started")).toBeDefined();
    });
  });

  describe("trigger, feature request and web routing", () => {
    it("renders the webhook trigger setup card with the URL", () => {
      render(
        <ToolResult
          row={row(
            {
              message: "Webhook ready",
              webhook_url: "https://hooks.example.com/h1",
            },
            "setup_agent_webhook_trigger",
          )}
        />,
      );

      expect(screen.getByText("Webhook ready")).toBeDefined();
      expect(screen.getByText("https://hooks.example.com/h1")).toBeDefined();
      expect(screen.getByRole("button", { name: "Copy" })).toBeDefined();
    });

    it("renders the trigger card for manual setup without a URL", () => {
      render(
        <ToolResult
          row={row(
            { manual_setup_required: true, message: "Set it up manually" },
            "setup_agent_webhook_trigger",
          )}
        />,
      );

      expect(screen.getByText("Set it up manually")).toBeDefined();
    });

    it("renders a link card for a created feature request", () => {
      render(
        <ToolResult
          row={row(
            {
              issue_url: "https://linear.app/agpt/issue/OPEN-1",
              issue_title: "Add dark mode",
            },
            "create_feature_request",
          )}
        />,
      );

      expect(screen.getByText("Add dark mode")).toBeDefined();
      expect(screen.getByLabelText("Open link").getAttribute("href")).toBe(
        "https://linear.app/agpt/issue/OPEN-1",
      );
    });

    it("falls back to key/value output for feature requests without a URL", () => {
      render(
        <ToolResult
          row={row(
            { status: "created", identifier: "FR-9" },
            "create_feature_request",
          )}
        />,
      );

      expect(screen.getByText("Status")).toBeDefined();
      expect(screen.getByText("created")).toBeDefined();
      expect(screen.getByText("FR-9")).toBeDefined();
    });

    it("renders web search results with the clamped answer", () => {
      render(
        <ToolResult
          row={row(
            {
              answer: "Paris is the capital",
              results: [{ title: "Wiki", url: "https://en.wikipedia.org/x" }],
            },
            "web_search",
          )}
        />,
      );

      expect(screen.getByText("Paris is the capital")).toBeDefined();
      expect(screen.getByText("Wiki")).toBeDefined();
    });

    it("renders a plain answer when web search returns no results", () => {
      render(
        <ToolResult row={row({ answer: "Just an answer" }, "web_search")} />,
      );

      expect(screen.getByText("Just an answer")).toBeDefined();
    });
  });

  describe("terminal, todos and file routing", () => {
    it("renders bash executions as a terminal", () => {
      render(
        <ToolResult
          row={row({ stdout: "file.txt", exit_code: 2 }, "bash_exec", {
            command: "ls",
          })}
        />,
      );

      expect(screen.getByText("ls")).toBeDefined();
      expect(screen.getByText("file.txt")).toBeDefined();
      expect(screen.getByText("exit 2")).toBeDefined();
    });

    it("renders TodoWrite rows as a todo list", () => {
      render(
        <ToolResult
          row={row({ ok: true }, "TodoWrite", {
            todos: [
              { content: "Task A", status: "completed" },
              { content: "Task B", status: "in_progress" },
              { content: "Task C", status: "pending" },
            ],
          })}
        />,
      );

      expect(screen.getByText("Task A")).toBeDefined();
      expect(screen.getByText("Task B")).toBeDefined();
      expect(screen.getByText("Task C")).toBeDefined();
    });

    it("renders workspace file reads as a file card", () => {
      render(
        <ToolResult
          row={row(
            {
              size_bytes: 2048,
              mime_type: "text/plain",
              preview: "hello world",
            },
            "read_workspace_file",
            { path: "notes.txt" },
          )}
        />,
      );

      expect(screen.getByText("notes.txt")).toBeDefined();
      expect(screen.getByText("2.0 KB")).toBeDefined();
      expect(screen.getByText("hello world")).toBeDefined();
    });
  });

  describe("generic fallbacks", () => {
    it("renders plain string output as preformatted text", () => {
      render(<ToolResult row={row("plain text output")} />);

      expect(screen.getByText("plain text output")).toBeDefined();
    });

    it("renders the message when nothing but base fields remain", () => {
      render(
        <ToolResult row={row({ type: "done", message: "All finished" })} />,
      );

      expect(screen.getByText("All finished")).toBeDefined();
    });

    it("renders outputs arrays through the output list", () => {
      render(
        <ToolResult
          row={row({ outputs: [{ name: "summary", value: "S" }] })}
        />,
      );

      expect(screen.getByText("summary")).toBeDefined();
      expect(screen.getByText("S")).toBeDefined();
    });

    it("renders outputs dictionaries through the output list", () => {
      render(<ToolResult row={row({ outputs: { answer: ["A"] } })} />);

      expect(screen.getByText("answer")).toBeDefined();
      expect(screen.getByText("A")).toBeDefined();
    });

    it("renders a link card for outputs with a URL and status code", () => {
      render(
        <ToolResult
          row={row({
            url: "https://example.com/page",
            title: "Example page",
            status_code: 200,
          })}
        />,
      );

      expect(screen.getByText("Example page")).toBeDefined();
      expect(screen.getByText("200")).toBeDefined();
      expect(screen.getByLabelText("Open link").getAttribute("href")).toBe(
        "https://example.com/page",
      );
    });

    it("renders a size for link outputs reporting bytes", () => {
      render(
        <ToolResult
          row={row({ url: "https://example.com/file", bytes: 4096 })}
        />,
      );

      expect(screen.getByText("4.0 KB")).toBeDefined();
    });

    it("renders single boolean outputs as a status card", () => {
      render(<ToolResult row={row({ type: "done", ok: true })} />);

      expect(screen.getByText("Done")).toBeDefined();
    });

    it("renders single numeric outputs as a stat card", () => {
      render(<ToolResult row={row({ deleted_count: 5 })} />);

      expect(screen.getByText("5")).toBeDefined();
      expect(screen.getByText("deleted count")).toBeDefined();
    });

    it("renders single string-array outputs as chips", () => {
      render(<ToolResult row={row({ tags: ["alpha", "beta"] })} />);

      expect(screen.getByText("Tags")).toBeDefined();
      expect(screen.getByText("alpha")).toBeDefined();
      expect(screen.getByText("beta")).toBeDefined();
    });

    it("renders single status strings as a status pill", () => {
      render(<ToolResult row={row({ status: "RUNNING" })} />);

      expect(screen.getByText("Status")).toBeDefined();
      expect(screen.getByText("running")).toBeDefined();
    });

    it("falls back to key/value output for run_block payloads without a block", () => {
      render(<ToolResult row={row({ note: "queued" }, "run_block")} />);

      expect(screen.getByText("Note")).toBeDefined();
      expect(screen.getByText("queued")).toBeDefined();
    });

    it("falls back to the message for connect_integration without setup info", () => {
      render(
        <ToolResult
          row={row({ message: "Please connect" }, "connect_integration")}
        />,
      );

      expect(screen.getByText("Please connect")).toBeDefined();
    });

    it("falls back to key/value output for empty web search results", () => {
      render(<ToolResult row={row({ results: [] }, "web_search")} />);

      expect(screen.getByText("Results")).toBeDefined();
    });

    it("falls back to key/value pairs for multi-field outputs", () => {
      render(<ToolResult row={row({ region: "us-east", replicas: 3 })} />);

      expect(screen.getByText("Region")).toBeDefined();
      expect(screen.getByText("us-east")).toBeDefined();
      expect(screen.getByText("Replicas")).toBeDefined();
      expect(screen.getByText("3")).toBeDefined();
    });
  });
});
