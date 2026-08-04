import type { SampleEvent } from "./helpers";

const ASSISTANT_ID = "sample-assistant-1";
const ASSISTANT_ID_2 = "sample-assistant-2";

// Scales tool delays only; text/reasoning streaming stays at normal speed.
const TOOL_TIME_MULTIPLIER = 2.5;

function chunks(text: string): string[] {
  const words = text.split(" ");
  const out: string[] = [];
  for (let i = 0; i < words.length; i += 4) {
    out.push((i > 0 ? " " : "") + words.slice(i, i + 4).join(" "));
  }
  return out;
}

function streamText(
  text: string,
  startDelay = 300,
  messageId = ASSISTANT_ID,
): SampleEvent[] {
  return [
    { delay: startDelay, kind: "text-start", messageId },
    ...chunks(text).map(
      (delta): SampleEvent => ({
        delay: 80,
        kind: "text-delta",
        messageId,
        delta,
      }),
    ),
  ];
}

function streamReasoning(text: string, startDelay = 400): SampleEvent[] {
  return [
    { delay: startDelay, kind: "reasoning-start", messageId: ASSISTANT_ID },
    ...chunks(text).map(
      (delta): SampleEvent => ({
        delay: 70,
        kind: "reasoning-delta",
        messageId: ASSISTANT_ID,
        delta,
      }),
    ),
    { delay: 250, kind: "reasoning-done", messageId: ASSISTANT_ID },
  ];
}

function toolStart(
  toolCallId: string,
  tool: string,
  input: unknown,
  delay: number,
  messageId = ASSISTANT_ID,
): SampleEvent {
  return {
    delay: delay * TOOL_TIME_MULTIPLIER,
    kind: "tool-start",
    messageId,
    toolCallId,
    tool,
    input,
  };
}

function toolOutput(
  toolCallId: string,
  output: unknown,
  delay: number,
  messageId = ASSISTANT_ID,
): SampleEvent {
  return {
    delay: delay * TOOL_TIME_MULTIPLIER,
    kind: "tool-output",
    messageId,
    toolCallId,
    output,
  };
}

function toolError(
  toolCallId: string,
  errorText: string,
  delay: number,
): SampleEvent {
  return {
    delay: delay * TOOL_TIME_MULTIPLIER,
    kind: "tool-error",
    messageId: ASSISTANT_ID,
    toolCallId,
    errorText,
  };
}

export function buildSampleEvents(): SampleEvent[] {
  return [
    {
      delay: 0,
      kind: "user",
      id: "sample-user-1",
      text: "Build a short research brief on global EV sales in 2024.",
    },
    { delay: 300, kind: "status", message: "Reading your message…" },
    { delay: 2200, kind: "assistant-start", id: ASSISTANT_ID },
    ...streamText(
      "I'll research 2024 global EV sales and put together a short brief. Let me plan the work, gather sources, and compute the year-over-year numbers.",
    ),

    ...streamReasoning(
      "Need the headline 2024 total plus the top makers. IEA and Rho Motion are the usual sources — search both, verify with one more outlet, then compute YoY growth and write the brief.",
    ),
    toolStart(
      "t-todo-1",
      "TodoWrite",
      {
        todos: [
          {
            content: "Show plan checklist",
            status: "in_progress",
            activeForm: "Planning the research",
          },
          { content: "Run parallel web searches", status: "pending" },
          { content: "Compute YoY table and chart", status: "pending" },
          { content: "Write ev-brief.md", status: "pending" },
        ],
      },
      350,
    ),
    toolOutput(
      "t-todo-1",
      { type: "todo_write", message: "Todos updated" },
      600,
    ),
    toolStart(
      "t-search-1",
      "web_search",
      { query: "global EV sales 2024 total units" },
      300,
    ),
    toolStart(
      "t-search-2",
      "web_search",
      { query: "IEA Global EV Outlook 2024 sales figures" },
      250,
    ),
    toolOutput(
      "t-search-1",
      {
        type: "web_search",
        message: "Search complete",
        query: "global EV sales 2024 total units",
        results: [
          {
            title: "Rho Motion: EV sales hit 17.1M in 2024",
            url: "https://rhomotion.com/ev-sales-2024",
            snippet: "Final 2024 tally for global EV sales…",
          },
        ],
        search_requests: 1,
      },
      1300,
    ),
    toolOutput(
      "t-search-2",
      {
        type: "web_search",
        message: "Search complete",
        query: "IEA Global EV Outlook 2024 sales figures",
        results: [
          {
            title: "IEA Global EV Outlook 2024",
            url: "https://www.iea.org/reports/global-ev-outlook-2024",
            snippet: "The IEA's annual outlook on electric mobility…",
          },
        ],
        search_requests: 1,
      },
      600,
    ),
    toolStart(
      "t-fetch-404",
      "web_fetch",
      { url: "https://example.com/this-page-does-not-exist-404" },
      350,
    ),
    toolError("t-fetch-404", "404 Not Found — page does not exist", 900),
    toolStart(
      "t-fetch-2",
      "web_fetch",
      { url: "https://www.iea.org/reports/global-ev-outlook-2024" },
      400,
    ),
    toolOutput(
      "t-fetch-2",
      {
        type: "web_fetch",
        message: "Fetched page",
        url: "https://www.iea.org/reports/global-ev-outlook-2024",
        status_code: 200,
        content_type: "text/html",
        title: "Global EV Outlook 2024 – Analysis - IEA",
        content: "# Global EV Outlook 2024…",
        content_length: 48213,
        truncated: false,
      },
      1100,
    ),
    toolStart(
      "t-bash-1",
      "bash_exec",
      { command: "python compute_ev_table.py --years 2023,2024" },
      400,
    ),
    toolOutput(
      "t-bash-1",
      {
        type: "bash_exec",
        message: "Command finished",
        stdout: "BYD  3.02M → 4.27M  (+41%)",
        stderr: "",
        exit_code: 0,
        timed_out: false,
      },
      1400,
    ),
    toolStart(
      "t-write-1",
      "write_workspace_file",
      { filename: "ev-brief.md" },
      300,
    ),
    toolOutput(
      "t-write-1",
      {
        type: "workspace_file_written",
        message: "Wrote ev-brief.md",
        file_id: "f-77",
        name: "ev-brief.md",
        path: "ev-brief.md",
        mime_type: "text/markdown",
        size_bytes: 2048,
        download_url: "workspace://f-77#text/markdown",
      },
      550,
    ),

    { delay: 500, kind: "status", message: "Analyzing result…" },
    ...streamText(
      "Sources are in — the headline total is about 17.1M units. Rendering the growth chart and finishing the brief now.",
      2400,
    ),

    toolStart(
      "t-bash-2",
      "bash_exec",
      { command: "python render_chart.py --out ev-growth.png" },
      400,
    ),
    toolOutput(
      "t-bash-2",
      {
        type: "bash_exec",
        message: "Command finished",
        stdout: "wrote ev-growth.png (91.4 KB)",
        stderr: "",
        exit_code: 0,
        timed_out: false,
      },
      1300,
    ),
    toolStart(
      "t-todo-2",
      "TodoWrite",
      {
        todos: [
          { content: "Write ev-brief.md", status: "completed" },
          {
            content: "Summarize findings",
            status: "in_progress",
            activeForm: "Summarizing findings",
          },
        ],
      },
      300,
    ),
    toolOutput(
      "t-todo-2",
      { type: "todo_write", message: "Todos updated" },
      500,
    ),

    ...streamText(
      "I can also automate this going forward — setting up an agent that tracks EV sales monthly and messages you the delta.",
      450,
    ),

    toolStart(
      "t-docs-1",
      "search_docs",
      { query: "scheduled agent runs" },
      350,
    ),
    toolOutput(
      "t-docs-1",
      {
        type: "doc_search_results",
        message: "3 results",
        count: 3,
        query: "scheduled agent runs",
        results: [
          {
            title: "Scheduling agents",
            path: "platform/scheduling.md",
            section: "Cron schedules",
            snippet: "Run agents on a recurring cron schedule…",
            score: 0.92,
          },
          {
            title: "Agent triggers",
            path: "platform/triggers.md",
            section: "Webhook triggers",
            snippet: "Trigger agents from external events…",
            score: 0.81,
          },
        ],
      },
      800,
    ),
    toolStart(
      "t-findblock-1",
      "find_block",
      { query: "send discord message" },
      300,
    ),
    toolOutput(
      "t-findblock-1",
      {
        type: "block_list",
        message: "Found 2 blocks",
        count: 2,
        query: "send discord message",
        blocks: [
          {
            id: "b-123",
            name: "Send Discord Message",
            description: "Post a message to a Discord channel",
            categories: ["COMMUNICATION"],
            provider: "discord",
          },
          {
            id: "b-124",
            name: "Discord Webhook",
            description: "Send via a Discord webhook URL",
            categories: ["COMMUNICATION"],
            provider: "discord",
          },
        ],
      },
      900,
    ),
    toolStart(
      "t-integration-1",
      "connect_integration",
      { provider: "Discord" },
      300,
    ),
    toolOutput(
      "t-integration-1",
      {
        type: "setup_requirements",
        message: "Connect Discord to continue",
        setup_info: {
          agent_id: "connect_discord",
          agent_name: "Discord",
          user_readiness: { has_all_credentials: false, ready_to_run: false },
        },
      },
      700,
    ),
    toolStart(
      "t-ask-1",
      "ask_question",
      {
        questions: [
          {
            question: "Which channel should the digest go to?",
            keyword: "channel",
          },
          { question: "What time should it post?", keyword: "time" },
        ],
      },
      350,
    ),
    toolOutput(
      "t-ask-1",
      {
        type: "agent_builder_clarification_needed",
        message: "I need two details before wiring this up.",
        questions: [
          {
            question: "Which channel should the digest go to?",
            keyword: "channel",
            example: "#general, #ev-updates",
          },
          {
            question: "What time should it post?",
            keyword: "time",
            example: "9:00 AM IST",
          },
        ],
      },
      600,
    ),

    { delay: 600, kind: "await-user" },
    {
      delay: 400,
      kind: "status",
      message: "Entering building mode — loading the agent guide…",
    },
    { delay: 2600, kind: "assistant-start", id: ASSISTANT_ID_2 },
    ...streamText(
      "Perfect — #ev-updates at 9:00 AM IST. Wiring up the pipeline now.",
      300,
      ASSISTANT_ID_2,
    ),

    toolStart(
      "t-block-1",
      "run_block",
      { block_id: "b-123", input_data: { channel: "#ev-updates" } },
      350,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-block-1",
      {
        type: "block_output",
        message: "Block 'Send Discord Message' executed successfully",
        block_id: "b-123",
        block_name: "Send Discord Message",
        provider: "discord",
        outputs: { status: ["sent"], message_id: ["m-88"] },
        success: true,
      },
      900,
      ASSISTANT_ID_2,
    ),
    toolStart(
      "t-findagent-1",
      "find_agent",
      { query: "market research tracker" },
      300,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-findagent-1",
      {
        type: "agents_found",
        message: "Found 1 agent",
        count: 1,
        agents: [
          {
            id: "autogpt/ev-sales-tracker",
            name: "EV Sales Tracker",
            description: "Monthly EV market pulls",
            source: "marketplace",
            creator: "autogpt",
            runs: 940,
          },
        ],
      },
      850,
      ASSISTANT_ID_2,
    ),
    toolStart(
      "t-agent-1",
      "run_agent",
      { username_agent_slug: "autogpt/ev-sales-tracker" },
      350,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-agent-1",
      {
        type: "execution_started",
        message: "Execution started",
        execution_id: "exec-1",
        graph_id: "graph-77",
        graph_name: "EV Sales Tracker",
        library_agent_id: "lib-201",
        library_agent_link: "/library/agents/lib-201",
        status: "COMPLETED",
      },
      1400,
      ASSISTANT_ID_2,
    ),
    toolStart(
      "t-output-1",
      "view_agent_output",
      { execution_id: "exec-1" },
      300,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-output-1",
      {
        type: "agent_output",
        message: "Latest execution output",
        agent_name: "EV Sales Tracker",
        agent_id: "graph-77",
        total_executions: 1,
        execution: {
          execution_id: "exec-1",
          status: "COMPLETED",
          outputs: { report: ["17.1M units in 2024"] },
        },
      },
      700,
      ASSISTANT_ID_2,
    ),
    toolStart(
      "t-mcp-1",
      "run_mcp_tool",
      { server_url: "https://mcp.linear.app", tool_name: "create_issue" },
      300,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-mcp-1",
      {
        type: "mcp_tool_output",
        message: "MCP tool executed",
        server_url: "https://mcp.linear.app",
        tool_name: "create_issue",
        result: { identifier: "OPEN-3211" },
        success: true,
      },
      900,
      ASSISTANT_ID_2,
    ),
    toolStart(
      "t-memory-1",
      "memory_store",
      { name: "EV brief preferences", content: "User tracks EV sales monthly" },
      300,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-memory-1",
      {
        type: "memory_store",
        message: "Memory stored",
        memory_name: "EV brief preferences",
      },
      600,
      ASSISTANT_ID_2,
    ),
    toolStart(
      "t-followup-1",
      "schedule_followup",
      { message: "Monthly EV sales update", delay_seconds: 2592000 },
      300,
      ASSISTANT_ID_2,
    ),
    toolOutput(
      "t-followup-1",
      {
        type: "schedule_created",
        message: "Follow-up scheduled",
        schedule_id: "sched-1",
        next_run_time: "2026-09-01T09:00:00+05:30",
        is_recurring: true,
      },
      650,
      ASSISTANT_ID_2,
    ),

    // No status here on purpose — exercises the bare loader + timer, which is
    // what a silent gap the backend hasn't labelled actually looks like.
    ...streamText(
      "Here's the brief. **Global EV sales reached ~17.1M units in 2024, up ~25% year-over-year.** China led with ~11M units, Europe was roughly flat at ~3M, and North America grew ~9% to ~1.8M. BYD and Tesla stayed the top two makers, with BYD widening its lead on plug-in hybrids. Full details are in ev-brief.md, and I've scheduled a monthly follow-up that posts the delta to your Discord.",
      2400,
      ASSISTANT_ID_2,
    ),
  ];
}
