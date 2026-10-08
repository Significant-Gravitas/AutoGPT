import type { UIDataTypes, UIMessage, UITools } from "ai";
import { TOOL_CARD_TOOLS } from "../components/ToolChain/ToolResult";
import {
  COMPACTION_PART_TYPE,
  EXPERT_CHANGE_TOOLS,
} from "../components/ToolChain/helpers";
import { EXPERT_ONBOARDING_PART_TYPE } from "../components/ExpertOnboardingCard/helpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;
type Part = Message["parts"][number];

export interface CardFixture {
  part: Part;
  /** Visible text that proves the card itself rendered, not a fallback. */
  marker: string;
}

/** Every part type the main chat has a card for, read from the renderer's own
 *  registries: the tool chain's card switch, plus the parts that render as
 *  cards of their own outside a chain. */
export const MAIN_CHAT_CARD_TYPES: string[] = [
  ...TOOL_CARD_TOOLS.map((tool) => `tool-${tool}`),
  COMPACTION_PART_TYPE,
  EXPERT_ONBOARDING_PART_TYPE,
  ...[...EXPERT_CHANGE_TOOLS].map((tool) => `tool-${tool}`),
];

function toolPart(
  type: string,
  output: unknown,
  input: Record<string, unknown> = {},
): Part {
  return {
    type,
    toolCallId: `call-${type.slice("tool-".length)}`,
    state: "output-available",
    input,
    output,
  } as unknown as Part;
}

function card(
  tool: string,
  output: unknown,
  marker: string,
  input?: Record<string, unknown>,
): [string, CardFixture] {
  const type = `tool-${tool}`;
  return [type, { part: toolPart(type, output, input), marker }];
}

function subSession(response: string, elapsed: number) {
  return {
    status: "COMPLETED",
    response,
    elapsed_seconds: elapsed,
    sub_autopilot_session_link: "/copilot?session=sub-1",
  };
}

function expertProposal(kind: string, name: string, role: string) {
  return {
    type: "expert_change_proposed",
    applied: false,
    confirmation_id: `confirm-${kind}`,
    preview: { kind, name, role },
  };
}

const DESKTOP = {
  type: "desktop_stream",
  message: "Desktop started.",
  desktop_stream: {
    kind: "desktop_stream",
    url: "https://6080-sbx.e2b.app/vnc.html?autoconnect=true",
    provider: "e2b",
    sandbox_id: "sbx-1",
  },
};

export const CARD_FIXTURES: Record<string, CardFixture> = Object.fromEntries([
  card(
    "run_agent",
    {
      graph_name: "Research Agent",
      execution_id: "exec-1",
      status: "COMPLETED",
      library_agent_link: "/library/agents/lib-1",
    },
    "Research Agent",
  ),
  card(
    "schedule_agent",
    { execution_id: "exec-2", graph_id: "graph-9" },
    "creator/scraper",
    { username_agent_slug: "creator/scraper" },
  ),
  card(
    "view_agent_output",
    { outputs: [{ name: "report", value: "Weekly report ready" }] },
    "Weekly report ready",
  ),
  card(
    "create_agent",
    { suggested_goal: "Build a scraper", reason: "You asked for it" },
    "Build a scraper",
  ),
  card(
    "customize_agent",
    { agent_name: "Custom Agent", library_agent_link: "/library/agents/lib-3" },
    "Custom Agent",
  ),
  card(
    "edit_agent",
    {
      agent_name: "Edited Agent",
      graph_version: 2,
      library_agent_link: "/library/agents/lib-2",
      agent_page_link: "/build?flowID=graph-2",
    },
    "Edited Agent",
  ),
  card(
    "run_sub_session",
    subSession("Everything worked", 75),
    "Everything worked",
  ),
  card(
    "get_sub_session_result",
    subSession("The poll came back", 30),
    "The poll came back",
  ),
  card("delegate_to_expert", subSession("Delegated", 75), "1m 15s"),
  card("handoff_to_expert", subSession("Handed off", 125), "2m 5s"),
  card(
    "consult_teammate",
    { verdict: "approve", reason: "The plan holds up" },
    "Teammate checked",
  ),
  card(
    "find_agent",
    {
      agents: [
        {
          id: "store-1",
          source: "marketplace",
          name: "Store Agent",
          creator: "toran",
        },
      ],
    },
    "Store Agent",
  ),
  card(
    "find_library_agent",
    {
      agents: [
        { id: "lib-1", source: "library", name: "Lib Agent", creator: "abhi" },
      ],
    },
    "Lib Agent",
  ),
  card(
    "find_block",
    {
      blocks: [
        {
          name: "HTTP Request",
          description: "Makes HTTP requests",
          categories: ["NETWORK"],
        },
      ],
    },
    "Makes HTTP requests",
  ),
  card(
    "find_capability",
    {
      capabilities: [
        { name: "Slack poster", purpose: "Posts to a channel", kind: "block" },
      ],
    },
    "Posts to a channel",
  ),
  card(
    "run_block",
    { block_name: "FillTextTemplateBlock", outputs: { output: ["Hello"] } },
    "Fill Text Template",
  ),
  card(
    "continue_run_block",
    {
      block_name: "Send Email",
      success: false,
      outputs: { error_message: ["SMTP down"] },
    },
    "SMTP down",
  ),
  card(
    "describe_capability",
    { block: { name: "Web Scraper", description: "Reads web pages" } },
    "Reads web pages",
  ),
  card(
    "run_capability",
    { block_name: "Summarizer", outputs: { summary: ["Short version"] } },
    "Short version",
  ),
  card(
    "resume_capability",
    { block_name: "Resumer", outputs: { status: ["Resumed fine"] } },
    "Resumed fine",
  ),
  card(
    "connect_integration",
    {
      type: "setup_requirements",
      message: "Connect GitHub to continue.",
      setup_info: {
        agent_id: "github",
        agent_name: "GitHub",
        requirements: {},
        user_readiness: { has_all_credentials: true },
      },
    },
    "GitHub",
  ),
  card(
    "decompose_goal",
    {
      steps: [
        { description: "Fetch the data", status: "completed" },
        { description: "Summarize it", status: "in_progress" },
      ],
    },
    "Fetch the data",
  ),
  card("validate_agent_graph", { valid: true }, "Graph is valid"),
  card(
    "fix_agent_graph",
    { valid_after_fix: true, fixes_applied: ["Linked input"] },
    "Fixed — applied 1 fix",
  ),
  card(
    "ask_question",
    {
      type: "agent_builder_clarification_needed",
      message: "I need a bit more detail.",
      questions: [
        {
          question: "Which memories should I forget?",
          keyword: "memories",
          options: ["Emberline", "Monday summaries"],
        },
      ],
    },
    "Which memories should I forget?",
  ),
  card(
    "list_schedules",
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
    "Daily run",
  ),
  card(
    "schedule_followup",
    { next_run_time: "2026-08-21T10:00:00Z", is_recurring: true },
    "Follow-up scheduled",
  ),
  card(
    "list_folders",
    { folders: [{ name: "Marketing", agent_count: 3 }] },
    "Marketing",
  ),
  card("create_folder", { folder: { name: "New Folder" } }, "New Folder"),
  card(
    "update_folder",
    { folder: { name: "Renamed Folder" } },
    "Renamed Folder",
  ),
  card("move_folder", { folder: { name: "Moved Folder" } }, "Moved Folder"),
  card(
    "list_workspace_files",
    {
      files: [{ path: "chart.png", mime_type: "image/png", size_bytes: 2048 }],
    },
    "chart.png",
  ),
  card(
    "search_docs",
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
    "How blocks work",
  ),
  card("get_doc_page", { title: "Getting Started" }, "Getting Started"),
  card(
    "setup_agent_webhook_trigger",
    { message: "Webhook ready", webhook_url: "https://hooks.example.com/h1" },
    "Webhook ready",
  ),
  card(
    "search_feature_requests",
    {
      results: [
        { title: "Dark mode", description: "Please", identifier: "FR-1" },
      ],
    },
    "Dark mode",
  ),
  card(
    "create_feature_request",
    {
      issue_url: "https://linear.app/agpt/issue/OPEN-1",
      issue_title: "Add dark mode",
    },
    "Add dark mode",
  ),
  card(
    "run_mcp_tool",
    { result: "The MCP tool answered" },
    "The MCP tool answered",
  ),
  card(
    "store_skill",
    {
      name: "Weekly digest",
      description: "Summarizes the week",
      triggers: ["every friday"],
    },
    "Weekly digest",
  ),
  card(
    "read_skill",
    { name: "Reader skill", description: "Reads things" },
    "Reader skill",
  ),
  card(
    "delete_skill",
    { name: "Old skill", description: "Not needed" },
    "Old skill",
  ),
  card(
    "list_skills",
    { skills: [{ name: "summarize" }, { name: "draft" }] },
    "summarize",
  ),
  card(
    "list_chat_platform_channels",
    { channels: [{ name: "general" }] },
    "#general",
  ),
  card(
    "web_search",
    {
      answer: "Paris is the capital",
      results: [{ title: "Wiki", url: "https://en.wikipedia.org/x" }],
    },
    "Paris is the capital",
  ),
  card("bash_exec", { stdout: "file.txt", exit_code: 0 }, "file.txt", {
    command: "ls",
  }),
  card("start_desktop", DESKTOP, "Desktop"),
  card("TodoWrite", { ok: true }, "Task A", {
    todos: [
      { content: "Task A", status: "completed" },
      { content: "Task B", status: "in_progress" },
    ],
  }),
  card(
    "read_workspace_file",
    { size_bytes: 2048, mime_type: "text/plain", preview: "hello world" },
    "notes.txt",
    { path: "notes.txt" },
  ),
  card(
    "write_workspace_file",
    { size_bytes: 7, mime_type: "text/plain" },
    "out.txt",
    { path: "out.txt", content: "written" },
  ),
  card("delete_workspace_file", { success: true }, "old.txt", {
    path: "old.txt",
  }),
  card("Read", "print(1)", "reader.py", { file_path: "/tmp/reader.py" }),
  card("Write", "ok", "writer.py", {
    file_path: "/tmp/writer.py",
    content: "x = 1",
  }),
  [
    COMPACTION_PART_TYPE,
    {
      part: toolPart(COMPACTION_PART_TYPE, {
        type: "context_compaction",
        messages_before: 40,
        messages_after: 8,
      }),
      marker: "Condensed",
    },
  ],
  [
    EXPERT_ONBOARDING_PART_TYPE,
    {
      part: toolPart(EXPERT_ONBOARDING_PART_TYPE, {
        type: "expert_onboarding",
        message: "Which outcome should I start with?",
        expert_id: "expert-zara",
        greeting: "Hi, I'm Zara.",
        steps: [
          {
            question: "Which outcome should I start with?",
            keyword: "outcome",
            options: ["Positioning", "Pricing"],
          },
        ],
      }),
      marker: "Which outcome should I start with?",
    },
  ],
  card("hire_expert", expertProposal("hire", "Zara", "Designer"), "Designer"),
  card("raise_expert", expertProposal("raise", "Otto", "Engineer"), "Engineer"),
  card(
    "update_expert",
    expertProposal("update", "Mira", "Researcher"),
    "Researcher",
  ),
  card(
    "confirm_expert_change",
    expertProposal("hire", "Theo", "Analyst"),
    "Analyst",
  ),
]);

/** One turn: the user asks, and the assistant's reply is the card under test.
 *  As the last message it is also live — an unanswered question or a fresh
 *  onboarding card is still waiting on the user. */
export function transcriptFor(fixture: CardFixture): Message[] {
  return [
    {
      id: "parity-user-1",
      role: "user",
      parts: [{ type: "text", text: "Go ahead" }],
    },
    {
      id: "parity-assistant-1",
      role: "assistant",
      parts: [fixture.part],
    },
  ];
}
