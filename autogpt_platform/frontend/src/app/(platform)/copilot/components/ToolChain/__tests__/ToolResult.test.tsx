import { render, screen, within } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, onTestFinished, vi } from "vitest";
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
    const pending = {
      dockId: "dock-1",
      questions: [{ question: "Which region?", keyword: "region" }],
      callIds: ["tool"],
    };

    it("renders the interactive answer form for the lifted pending row", () => {
      render(
        <PendingQuestionsContext.Provider value={pending}>
          <ToolResult
            row={{
              ...row({ questions: [{ question: "Which region?" }] }),
              lifted: true,
            }}
          />
        </PendingQuestionsContext.Provider>,
      );

      expect(screen.getByText("questions-form:dock-1")).toBeDefined();
    });

    // A running row animating out of the chain as its output lands is still
    // mounted beside its lifted copy; a second form there would register the
    // same card twice (SECRT-3135).
    it("leaves the form to the lifted copy of the pending row", () => {
      render(
        <PendingQuestionsContext.Provider value={pending}>
          <ToolResult
            row={row({ questions: [{ question: "Which region?" }] })}
          />
        </PendingQuestionsContext.Provider>,
      );

      expect(screen.queryByText("questions-form:dock-1")).toBeNull();
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

      expect(screen.getByText("Expert")).toBeDefined();
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
      expect(screen.getByLabelText("Open workflow").getAttribute("href")).toBe(
        "/library/agents/lib-1",
      );
    });
  });

  describe("block routing", () => {
    it.each([
      { block_name: "FillTextTemplateBlock", outputs: { output: ["Hello"] } },
      { block: { name: "FillTextTemplateBlock" } },
    ])("normalizes expanded block names from %j", (output) => {
      render(<ToolResult row={row(output, "run_block")} />);
      expect(screen.getByText("Fill Text Template")).toBeDefined();
      expect(screen.queryByText("FillTextTemplateBlock")).toBeNull();
    });

    it("keeps unknown block output cards generic instead of showing a UUID", () => {
      render(
        <ToolResult
          row={row(
            {
              block_id: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
              outputs: { output: ["Hello"] },
            },
            "run_block",
          )}
        />,
      );
      expect(screen.getByText("Block")).toBeDefined();
      expect(
        screen.queryByText("db7d8f02-2f44-4c55-ab7a-eae0941f0c30"),
      ).toBeNull();
    });

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

      expect(screen.getByText("Workflow is valid")).toBeDefined();
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
      expect(screen.getByText("3 workflows")).toBeDefined();
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
    it("keeps technical details closed for an approved call, showing only the command approved", () => {
      const csv = "date,channel\n2026-11-10,Instagram\n2026-11-13,Shopify";
      render(
        <ToolResult
          row={{
            ...row({ stdout: csv, exit_code: 0 }, "bash_exec", {
              command: "cat calendar.csv",
            }),
            held: {
              state: "approved",
              reviewId: "copilot-node-gate-bash_exec:a1",
            },
          }}
        />,
      );

      expect(
        screen
          .getByRole("button", { name: "Show technical details" })
          .getAttribute("aria-expanded"),
      ).toBe("false");
      expect(screen.getByText("You approved this command")).toBeDefined();
      expect(screen.getByText("cat calendar.csv")).toBeDefined();
      expect(screen.queryByText(/2026-11-10,Instagram/)).toBeNull();
    });

    it("keeps a bash step's command and output behind technical details", async () => {
      const user = userEvent.setup();
      render(
        <ToolResult
          row={row({ stdout: "file.txt", exit_code: 2 }, "bash_exec", {
            command: "ls",
          })}
        />,
      );

      expect(screen.queryByText("ls")).toBeNull();
      expect(screen.queryByText("file.txt")).toBeNull();
      const toggle = screen.getByRole("button", {
        name: "Show technical details",
      });
      expect(toggle.getAttribute("aria-expanded")).toBe("false");

      await user.click(toggle);

      expect(toggle.getAttribute("aria-expanded")).toBe("true");
      expect(screen.getByText("ls")).toBeDefined();
      expect(screen.getByText("file.txt")).toBeDefined();
      expect(screen.getByText("exit 2")).toBeDefined();
    });

    it("says plainly that a bash step failed", () => {
      render(
        <ToolResult
          row={row({ stdout: "", stderr: "boom", exit_code: 1 }, "bash_exec", {
            command: "false",
          })}
        />,
      );

      expect(screen.getByText("This step didn't work")).toBeDefined();
      expect(screen.queryByText("boom")).toBeNull();
    });

    it("shows both stdout and stderr, including a timeout note, under technical details", async () => {
      const user = userEvent.setup();
      render(
        <ToolResult
          row={row(
            {
              stdout: "partial rows",
              stderr: "Timed out after 10s\nboom",
              exit_code: -1,
              timed_out: true,
            },
            "bash_exec",
            { command: "slow-export" },
          )}
        />,
      );

      await user.click(
        screen.getByRole("button", { name: "Show technical details" }),
      );
      expect(screen.getByText("partial rows")).toBeDefined();
      expect(screen.getByText(/Timed out after 10s/)).toBeDefined();
    });

    it.each([
      ["curl -fs https://shop.example/api/orders", 22, ""],
      ["./check_stock.sh", 2, "ERROR: stock file missing"],
      ["grep -c TBD brief.md missing.md", 2, ""],
      ["grep -q ready status.txt && ./publish.sh", 1, ""],
      ["grep -q x orders.csv || ./fallback.sh", 1, ""],
      ["./build.sh || grep -q x out.log", 1, ""],
      ["cd drafts 2>/dev/null && test -f brief.md", 1, ""],
      ["set -o pipefail; ./build.sh | grep -q x out.log", 1, ""],
    ])(
      "calls a non-zero exit a failure even with nothing on stderr (%s)",
      (command, exitCode, stdout) => {
        render(
          <ToolResult
            row={row({ stdout, stderr: "", exit_code: exitCode }, "bash_exec", {
              command,
            })}
          />,
        );
        expect(screen.getByText("This step didn't work")).toBeDefined();
      },
    );

    it.each([
      "grep -q TBD brief.md",
      "test -f brief.md",
      "[ -d drafts ]",
      "cat a.txt | grep -q x",
      "diff old.csv new.csv",
      "grep -q TBD brief.md && echo found",
      "test -f brief.md && wc -l brief.md",
    ])(
      "does not call an expected 'no' (exit 1 from %s) a failure",
      (command) => {
        render(
          <ToolResult
            row={row({ stdout: "", stderr: "", exit_code: 1 }, "bash_exec", {
              command,
            })}
          />,
        );
        expect(screen.queryByText("This step didn't work")).toBeNull();
      },
    );

    it("does not call a quiet non-zero exit a failure (grep -q, test -f)", async () => {
      const user = userEvent.setup();
      render(
        <ToolResult
          row={row({ stdout: "", stderr: "", exit_code: 1 }, "bash_exec", {
            command: "grep -q TBD brief.md",
          })}
        />,
      );

      expect(screen.queryByText("This step didn't work")).toBeNull();
      await user.click(
        screen.getByRole("button", { name: "Show technical details" }),
      );
      expect(screen.getByText("exit 1")).toBeDefined();
    });

    it("says plainly that a bash step that timed out failed", () => {
      render(
        <ToolResult
          row={row(
            { stdout: "", stderr: "", exit_code: -1, timed_out: true },
            "bash_exec",
            { command: "sleep 999" },
          )}
        />,
      );

      expect(screen.getByText("This step didn't work")).toBeDefined();
    });

    it("says plainly that a bash step the tool refused failed, with its message under details", async () => {
      const user = userEvent.setup();
      render(
        <ToolResult
          row={row(
            {
              type: "error",
              message: "No command provided.",
              error: "empty_command",
            },
            "bash_exec",
            { command: "" },
          )}
        />,
      );

      expect(screen.getByText("This step didn't work")).toBeDefined();
      expect(screen.queryByText("No command provided.")).toBeNull();
      await user.click(
        screen.getByRole("button", { name: "Show technical details" }),
      );
      expect(screen.getByText("No command provided.")).toBeDefined();
    });

    it("shows a result stored as a JSON string once, without its plumbing fields", async () => {
      const user = userEvent.setup();
      const stored = JSON.stringify({
        type: "bash_exec",
        message: "Command executed with status code 0",
        session_id: "sess-123",
        stdout: "launch notes ready",
        stderr: "",
        exit_code: 0,
      });
      const { container } = render(
        <ToolResult
          row={row(stored, "bash_exec", { command: "cat notes.txt" })}
        />,
      );

      await user.click(
        screen.getByRole("button", { name: "Show technical details" }),
      );
      expect(screen.getAllByText("launch notes ready")).toHaveLength(1);
      expect(container.textContent).not.toContain("sess-123");
      expect(container.textContent).not.toContain("Command executed");
    });

    it("keeps an oversized bash result's preview under technical details", async () => {
      const user = userEvent.setup();
      const envelope = [
        '<tool-output-truncated total_chars=90000 workspace_path="tool-outputs/x.json">',
        '{"type":"bash_exec","stdout":"date,channel',
        "Full output (90000 chars) saved to the workspace.",
        "</tool-output-truncated>",
      ].join("\n");
      render(
        <ToolResult
          row={row(envelope, "bash_exec", { command: "cat big.csv" })}
        />,
      );

      expect(screen.queryByText(/date,channel/)).toBeNull();
      await user.click(
        screen.getByRole("button", { name: "Show technical details" }),
      );
      expect(screen.getByText("cat big.csv")).toBeDefined();
      expect(screen.getByText(/date,channel/)).toBeDefined();
      expect(screen.getByText(/Preview of 90,000 characters/)).toBeDefined();
    });

    it.each([
      [3, true],
      [0, false],
    ])(
      "reads an oversized bash result's exit %i from what was kept",
      (exitCode, failed) => {
        const tail = `"stderr": "", "exit_code": ${exitCode}, "timed_out": false}`;
        const envelope = [
          '<tool-output-truncated total_chars=120000 workspace_path="tool-outputs/x.json">',
          `{"type": "bash_exec", "message": "Command executed", "stdout": "ERROR: export… (omitted 110000 chars)…rows done", ${tail}`,
          "Full output (120,000 chars) saved to workspace.",
          "</tool-output-truncated>",
        ].join("\n");
        render(
          <ToolResult
            row={row(envelope, "bash_exec", { command: "./export.sh" })}
          />,
        );

        expect(screen.queryByText("This step didn't work") !== null).toBe(
          failed,
        );
      },
    );

    it("previews CSV printed by a bash step as a table, with the file to download", async () => {
      const user = userEvent.setup();
      const csv = [
        "date,channel,post",
        "2026-11-10,Instagram,Teaser",
        "2026-11-13,Instagram,Ingredient spotlight",
        "2026-11-17,Shopify,Waitlist page",
        "2026-11-25,Shopify,Launch day",
      ].join("\n");
      const created: Blob[] = [];
      const createObjectURL = vi.fn((blob: Blob) => {
        created.push(blob);
        return "blob:preview";
      });
      const realCreate = URL.createObjectURL;
      const realRevoke = URL.revokeObjectURL;
      URL.createObjectURL = createObjectURL;
      const revokeObjectURL = vi.fn();
      URL.revokeObjectURL = revokeObjectURL;
      const timers = vi.spyOn(window, "setTimeout");
      // A real browser downloads the blob; the test DOM would navigate to it.
      const clicked = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(() => {});
      onTestFinished(() => {
        URL.createObjectURL = realCreate;
        URL.revokeObjectURL = realRevoke;
        clicked.mockRestore();
        timers.mockRestore();
      });
      render(
        <ToolResult
          row={row({ stdout: csv, exit_code: 0 }, "bash_exec", {
            command: "cat calendar.csv",
          })}
        />,
      );

      const table = screen.getByRole("table", { name: "Table preview" });
      expect(
        within(table).getByRole("columnheader", { name: "channel" }),
      ).toBeDefined();
      expect(
        within(table).getByRole("cell", { name: "Ingredient spotlight" }),
      ).toBeDefined();
      expect(
        within(table).queryByRole("cell", { name: "Launch day" }),
      ).toBeNull();
      expect(screen.getByText("4 rows · date, channel, post")).toBeDefined();
      expect(screen.getByText("+1 more row")).toBeDefined();
      expect(screen.queryByText(/2026-11-13,Instagram/)).toBeNull();

      await user.click(screen.getByRole("button", { name: "Download CSV" }));
      expect(createObjectURL).toHaveBeenCalledTimes(1);
      expect(await created[0].text()).toBe(csv);
      // Revoked only after the download has had time to start.
      expect(revokeObjectURL).not.toHaveBeenCalled();
      const revoke = timers.mock.calls.find(([, delay]) => (delay ?? 0) > 0);
      expect(revoke).toBeDefined();
      (revoke?.[0] as () => void)();
      expect(revokeObjectURL).toHaveBeenCalledWith("blob:preview");
      expect(clicked).toHaveBeenCalledTimes(1);
      expect((clicked.mock.contexts[0] as HTMLAnchorElement).download).toBe(
        "table.csv",
      );
    });

    it("never previews a path, command or id held in a table's cells", () => {
      const csv = [
        "date,source,step,ref,status",
        "2026-11-10,~/workspace/dewleaf/brief.md,sed -i s/TBD/25/ brief.md,2616d1f1-16bb-43c0-9f0e-6a7c2a9d0b11,done",
        "2026-11-13,~/workspace/dewleaf/plan.md,cat plan,7a1b2c3d-16bb-43c0-9f0e-6a7c2a9d0b11,draft",
        "2026-11-17,~/workspace/dewleaf/calendar.csv,git status,9f8e7d6c-16bb-43c0-9f0e-6a7c2a9d0b11,draft",
      ].join("\n");
      const { container } = render(
        <ToolResult
          row={row({ stdout: csv, exit_code: 0 }, "bash_exec", {
            command: "cat log.csv",
          })}
        />,
      );

      const table = screen.getByRole("table", { name: "Table preview" });
      const headers = within(table)
        .getAllByRole("columnheader")
        .map((th) => th.textContent);
      expect(headers).toEqual(["date", "status"]);
      expect(
        within(table).getAllByRole("cell", { name: "draft" }),
      ).toHaveLength(2);
      for (const leak of [
        "~/workspace",
        "sed -i",
        "git status",
        "cat plan",
        "2616d1f1",
      ])
        expect(container.textContent).not.toContain(leak);
    });

    it("keeps only the download when every column of a table is technical", () => {
      const csv = [
        "path,command",
        "~/workspace/a/brief.md,sed -i s/a/b/ brief.md",
        "~/workspace/a/plan.md,cat plan",
        "~/workspace/a/notes.md,git status",
      ].join("\n");
      const { container } = render(
        <ToolResult
          row={row({ stdout: csv, exit_code: 0 }, "bash_exec", {
            command: "cat paths.csv",
          })}
        />,
      );

      expect(screen.queryByRole("table")).toBeNull();
      expect(screen.getByText("3 rows · path, command")).toBeDefined();
      expect(
        screen.getByRole("button", { name: "Download CSV" }),
      ).toBeDefined();
      expect(container.textContent).not.toContain("~/workspace");
      expect(container.textContent).not.toContain("sed -i");
    });

    it("never previews a JSON field whose value is a path, command or id", () => {
      const json = JSON.stringify({
        launch: "2026-11-25",
        brief: "~/workspace/dewleaf/brief.md",
        next_step: "cd ~/workspace && cat brief.md",
        run: "2616d1f1-16bb-43c0-9f0e-6a7c2a9d0b11",
        check: "git status",
        price: 899,
      });
      const { container } = render(
        <ToolResult
          row={row({ stdout: json, exit_code: 0 }, "bash_exec", {
            command: "cat brief.json",
          })}
        />,
      );

      expect(screen.getByText("2026-11-25")).toBeDefined();
      expect(screen.getByText("899")).toBeDefined();
      for (const leak of [
        "~/workspace",
        "cat brief.md",
        "2616d1f1",
        "git status",
      ])
        expect(container.textContent).not.toContain(leak);
    });

    it("previews JSON printed by a bash step field by field, never as raw JSON", () => {
      render(
        <ToolResult
          row={row(
            { stdout: '{"launch": "2026-11-25", "price": 899}', exit_code: 0 },
            "bash_exec",
            { command: "cat brief.json" },
          )}
        />,
      );

      expect(screen.getByText("Data")).toBeDefined();
      expect(screen.getByText("2 fields · launch, price")).toBeDefined();
      expect(screen.getByText("2026-11-25")).toBeDefined();
      expect(screen.getByText("899")).toBeDefined();
      expect(screen.queryByText(/"launch"/)).toBeNull();
      expect(
        screen.getByRole("button", { name: "Download JSON" }),
      ).toBeDefined();
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

      expect(screen.getByText("Notes")).toBeDefined();
      expect(screen.getByText("2.0 KB")).toBeDefined();
      expect(screen.getByText("hello world")).toBeDefined();
    });

    it("keeps a file's path behind technical details", async () => {
      const user = userEvent.setup();
      const path = "~/workspace/dewleaf/launch_brief.md";
      render(
        <ToolResult
          row={row({ size_bytes: 512 }, "write_workspace_file", { path })}
        />,
      );

      expect(screen.getByText("Launch Brief")).toBeDefined();
      expect(screen.queryByText(path)).toBeNull();
      await user.click(
        screen.getByRole("button", { name: "Show technical details" }),
      );
      expect(screen.getByText(path)).toBeDefined();
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
      expect(
        screen.getByLabelText("Open link").closest(".ph-no-capture"),
      ).toBeNull();
    });

    it("keeps an update_preset webhook URL out of session replays", () => {
      const url =
        "https://backend.agpt.co/api/integrations/generic_webhook/webhooks/wh-1/ingress";
      render(
        <ToolResult
          row={row(
            {
              type: "preset_updated",
              message: "Preset updated",
              preset_id: "p1",
              name: "My Trigger",
              is_active: true,
              webhook_url: url,
            },
            "update_preset",
          )}
        />,
      );

      expect(screen.getByText(url).closest(".ph-no-capture")).not.toBeNull();
      expect(
        screen.getByLabelText("Open link").closest(".ph-no-capture"),
      ).not.toBeNull();
    });

    it("blocks a webhook_url link card even off the ingress path", () => {
      render(
        <ToolResult
          row={row({ webhook_url: "https://hooks.example.com/h1" })}
        />,
      );

      expect(
        screen.getByLabelText("Open link").closest(".ph-no-capture"),
      ).not.toBeNull();
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

    it("says an empty web search found nothing", () => {
      render(<ToolResult row={row({ results: [] }, "web_search")} />);

      expect(screen.getByText("Nothing found.")).toBeDefined();
      expect(document.body.textContent).not.toContain("[]");
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

describe("ToolResult start_desktop", () => {
  afterEach(() => cleanup());

  it("embeds the live desktop stream instead of dumping the payload", () => {
    render(
      <ToolResult
        row={row(
          {
            type: "desktop_stream",
            message: "Desktop started.",
            desktop_stream: {
              kind: "desktop_stream",
              url: "https://6080-sbx.e2b.app/vnc.html?autoconnect=true",
              provider: "e2b",
              sandbox_id: "sbx-1",
            },
          },
          "start_desktop",
        )}
      />,
    );

    const frame = screen.getByTitle("Interactive desktop (sbx-1)");
    expect(frame.getAttribute("src")).toBe(
      "https://6080-sbx.e2b.app/vnc.html?autoconnect=true",
    );
    expect(screen.getByText("Open in new tab")).toBeDefined();
    expect(screen.queryByText(/"kind"/)).toBeNull();
  });

  it("shows a shared-transcript viewer a notice, not the owner-only frame", () => {
    render(
      <ToolResult
        readOnly
        row={row(
          {
            type: "desktop_stream",
            message: "Desktop started.",
            desktop_stream: {
              kind: "desktop_stream",
              url: "/api/proxy/api/desktop-preview?token=abc",
              provider: "e2b",
              sandbox_id: "sbx-1",
              requires_auth: true,
            },
          },
          "start_desktop",
        )}
      />,
    );

    expect(screen.queryByTitle(/Interactive desktop/)).toBeNull();
    expect(screen.getByText(/only visible to the owner/i)).toBeDefined();
  });

  it("falls back to the generic view when the stream is missing", () => {
    render(
      <ToolResult
        row={row({ message: "Failed to start the desktop." }, "start_desktop")}
      />,
    );
    expect(screen.getByText("Failed to start the desktop.")).toBeDefined();
    expect(screen.queryByTitle(/Interactive desktop/)).toBeNull();
  });
});

describe("ToolResult run_capability", () => {
  afterEach(() => cleanup());

  const desktopPayload = {
    type: "desktop_stream",
    message: "Desktop started.",
    desktop_stream: {
      kind: "desktop_stream",
      url: "/api/proxy/api/desktop-preview?token=abc",
      provider: "e2b",
      sandbox_id: "sbx-2",
    },
  };

  it("embeds the desktop when start_desktop ran as a capability", () => {
    render(
      <ToolResult
        row={row(desktopPayload, "run_capability", {
          id: "tool:start_desktop",
          input: {},
        })}
      />,
    );

    const frame = screen.getByTitle("Interactive desktop (sbx-2)");
    expect(frame.getAttribute("src")).toBe(
      "/api/proxy/api/desktop-preview?token=abc",
    );
    expect(screen.queryByText(/"kind"/)).toBeNull();
    expect(screen.queryByText(/desktop stream/i)).toBeNull();
  });

  it("keeps the shared-transcript notice on the capability path", () => {
    render(
      <ToolResult
        readOnly
        row={row(
          {
            ...desktopPayload,
            desktop_stream: {
              ...desktopPayload.desktop_stream,
              requires_auth: true,
            },
          },
          "resume_capability",
          { id: "tool:start_desktop" },
        )}
      />,
    );

    expect(screen.queryByTitle(/Interactive desktop/)).toBeNull();
    expect(screen.getByText(/only visible to the owner/i)).toBeDefined();
  });

  it("keeps the capability id behind technical details", async () => {
    const user = userEvent.setup();
    render(
      <ToolResult
        row={row(
          {
            type: "capability_details",
            block: { name: "SendWebRequestBlock", description: "Calls a URL" },
          },
          "describe_capability",
          { id: "block:6f1c2d3e" },
        )}
      />,
    );

    expect(screen.queryByText(/block:6f1c2d3e/)).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "Show technical details" }),
    );
    expect(screen.getByText("block:6f1c2d3e")).toBeDefined();
  });

  it("still renders a plain capability result as before", () => {
    render(
      <ToolResult
        row={row(
          { message: "Ran it.", result: { total: 3 } },
          "run_capability",
          { id: "tool:count" },
        )}
      />,
    );

    expect(screen.queryByTitle(/Interactive desktop/)).toBeNull();
    expect(screen.getByText(/total/i)).toBeDefined();
  });
});

describe("ToolResult deferred platform tools", () => {
  const cases: {
    tool: string;
    input: Record<string, unknown>;
    output: Record<string, unknown>;
    marker: string | RegExp;
  }[] = [
    {
      tool: "find_agent",
      input: { query: "leads" },
      output: { agents: [{ id: "lib-1", source: "library", name: "Leads" }] },
      marker: "Leads",
    },
    {
      tool: "find_library_agent",
      input: { query: "leads" },
      output: {
        agents: [{ id: "lib-2", source: "library", name: "Lib", runs: 1200 }],
      },
      marker: "1,200 runs",
    },
    {
      tool: "list_schedules",
      input: {},
      output: {
        schedules: [
          {
            name: "Daily run",
            next_run_time: "2026-08-21T10:00:00Z",
            cron: "0 10 * * *",
            kind: "copilot_turn",
          },
        ],
      },
      marker: "chat",
    },
    {
      tool: "list_folders",
      input: {},
      output: { folders: [{ name: "Marketing", agent_count: 3 }] },
      marker: "3 workflows",
    },
    {
      tool: "search_docs",
      input: { query: "blocks" },
      output: {
        results: [
          {
            title: "Blocks",
            section: "Guide",
            snippet: "How blocks work",
            doc_url: "https://docs.agpt.co/blocks",
          },
        ],
      },
      marker: "How blocks work",
    },
    {
      tool: "list_skills",
      input: {},
      output: { skills: [{ name: "summarize" }, { name: "draft" }] },
      marker: "Skills",
    },
    {
      tool: "get_sub_session_result",
      input: { sub_session_id: "sub-1" },
      output: {
        status: "COMPLETED",
        response: "Everything worked",
        elapsed_seconds: 75,
        sub_autopilot_session_link: "/copilot?session=sub-1",
      },
      marker: "1m 15s",
    },
    {
      tool: "validate_agent_graph",
      input: { agent_json: {} },
      output: { valid: true },
      marker: "Workflow is valid",
    },
    {
      tool: "setup_agent_webhook_trigger",
      input: { library_agent_id: "lib-1" },
      output: {
        message: "Webhook ready",
        webhook_url: "https://hooks.example.com/h1",
      },
      marker: "https://hooks.example.com/h1",
    },
    {
      tool: "setup_agent_webhook_trigger",
      input: { library_agent_id: "lib-1" },
      output: {
        message: "Connect an account",
        setup_info: { agent_id: "agent-1", agent_name: "Tracker" },
      },
      marker: "mode:trigger",
    },
  ];

  it.each(cases)(
    "renders the $tool card the same when it ran through run_capability",
    ({ tool, input, output, marker }) => {
      const direct = render(<ToolResult row={row(output, tool, input)} />);
      expect(screen.getByText(marker)).toBeDefined();
      const expected = direct.container.innerHTML;
      direct.unmount();

      const wrapped = render(
        <ToolResult
          row={row(output, "run_capability", { id: `tool:${tool}`, input })}
        />,
      );

      expect(wrapped.container.innerHTML).toBe(expected);
    },
  );

  it("keeps the open-agent link for a deferred find_agent", () => {
    render(
      <ToolResult
        row={row(
          { agents: [{ id: "lib-1", source: "library", name: "Leads" }] },
          "run_capability",
          { id: "tool:find_agent", input: { query: "leads" } },
        )}
      />,
    );

    expect(screen.getByLabelText("Open workflow").getAttribute("href")).toBe(
      "/library/agents/lib-1",
    );
  });

  it("resolves a bare tool name the way the backend does", () => {
    render(
      <ToolResult
        row={row({ valid: true }, "run_capability", {
          id: "validate_agent_graph",
          input: {},
        })}
      />,
    );

    expect(screen.getByText("Workflow is valid")).toBeDefined();
  });

  it("reads the nested input as the tool's input", () => {
    render(
      <ToolResult
        row={row(
          { execution_id: "exec-2", graph_id: "graph-9" },
          "run_capability",
          {
            id: "tool:schedule_agent",
            input: { username_agent_slug: "creator/scraper" },
          },
        )}
      />,
    );

    expect(screen.getByText("creator/scraper")).toBeDefined();
  });

  it.each(["delete_workspace_file", "store_skill", "validate_agent_graph"])(
    "shows the error when a deferred %s call failed",
    (tool) => {
      render(
        <ToolResult
          row={row(
            { type: "error", message: "File not found: notes.txt" },
            "run_capability",
            { id: `tool:${tool}`, input: { path: "notes.txt" } },
          )}
        />,
      );

      expect(screen.getByText("File not found: notes.txt")).toBeDefined();
    },
  );

  it("does not take an inherited object key for a tool name", () => {
    render(
      <ToolResult
        row={row({ note: "first", other: "second" }, "run_capability", {
          id: "constructor",
          input: { url: "https://nested.example.com/page" },
        })}
      />,
    );

    expect(screen.queryByText(/nested\.example\.com/)).toBeNull();
    expect(screen.getByText("first")).toBeDefined();
  });

  it("does not render a validate_only response as an execution card", () => {
    render(
      <ToolResult
        row={row({ valid: true }, "run_capability", {
          id: "tool:validate_agent_graph",
          input: {},
          validate_only: true,
        })}
      />,
    );

    expect(screen.queryByText("Workflow is valid")).toBeNull();
  });

  it("does not render capability details as an execution card", () => {
    render(
      <ToolResult
        row={row(
          {
            type: "capability_details",
            message: "Validate a graph.",
            capability: { id: "tool:validate_agent_graph", kind: "tool" },
            valid: true,
          },
          "run_capability",
          { id: "tool:validate_agent_graph", input: {} },
        )}
      />,
    );

    expect(screen.queryByText("Workflow is valid")).toBeNull();
  });

  it("does not render a describe_capability response as an execution card", () => {
    render(
      <ToolResult
        row={row({ valid: true }, "describe_capability", {
          id: "tool:validate_agent_graph",
        })}
      />,
    );

    expect(screen.queryByText("Workflow is valid")).toBeNull();
  });

  it("leaves a resumed call alone, since it names a review and not a tool", () => {
    render(
      <ToolResult
        row={row({ valid: true }, "resume_capability", {
          review_id: "copilot-node-abc:1f",
        })}
      />,
    );

    expect(screen.queryByText("Workflow is valid")).toBeNull();
  });

  it("keeps block capability rows on the block card", () => {
    render(
      <ToolResult
        row={row(
          {
            block_id: "b1",
            block_name: "Get Weather",
            outputs: { temperature: [21] },
          },
          "run_capability",
          { id: "block:b1", input: { city: "Oslo" } },
        )}
      />,
    );

    expect(screen.getByText("Get Weather")).toBeDefined();
  });

  it("keeps MCP capability rows on the MCP setup card", () => {
    render(
      <ToolResult
        row={row(
          {
            message: "Connect the tracker",
            setup_info: { agent_id: "mcp", agent_name: "Tracker" },
          },
          "run_capability",
          { id: "mcp:mcp.example.com", input: { tool: "list_issues" } },
        )}
      />,
    );

    expect(screen.getByText("mcp-setup-card")).toBeDefined();
  });
});
