import type { ToolUIPart } from "ai";
import type { MessagePart } from "../components/ChatMessagesContainer/helpers";

export interface SampleTool {
  tool?: string;
  reasoning?: string;
  reasoningStreaming?: boolean;
  state?: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
  errorText?: string;
}

export function toPart(sample: SampleTool, index: number): MessagePart {
  if (sample.reasoning !== undefined) {
    return {
      type: "reasoning",
      text: sample.reasoning,
      state: sample.reasoningStreaming ? "streaming" : "done",
    } as MessagePart;
  }
  return {
    type: `tool-${sample.tool}`,
    toolCallId: `demo-${sample.tool}-${index}`,
    state: sample.state ?? "output-available",
    input: sample.input,
    output: sample.output,
    errorText: sample.errorText,
  } as MessagePart;
}

const DIFF_SAMPLE = `@@ -12,7 +12,9 @@ function verifyToken(token: string) {
-  return jwt.verify(token, secret);
+  return jwt.verify(token, secret, {
+    algorithms: ["HS256"],
+    issuer: "autogpt.co",
+  });
 }`;

// Outputs mirror the backend Pydantic tool responses
// (backend/copilot/tools/models.py): every payload carries the
// `type`/`message` envelope plus the exact domain field names the real
// stream delivers, so the playground renders what production renders.
export const CATALOG_SECTIONS: { title: string; tools: SampleTool[] }[] = [
  {
    title: "Agents",
    tools: [
      {
        tool: "find_agent",
        input: { query: "newsletter writer" },
        output: {
          type: "agents_found",
          message: "Found 2 agents matching 'newsletter writer'",
          title: "Available Agents",
          count: 2,
          agents: [
            {
              id: "autogpt/newsletter-studio",
              name: "Newsletter Studio",
              description: "Drafts and schedules weekly newsletters",
              source: "marketplace",
              creator: "autogpt",
              rating: 4.8,
              runs: 1832,
            },
            {
              id: "autogpt/digest-bot",
              name: "Digest Bot",
              description: "Turns RSS feeds into a morning digest",
              source: "marketplace",
              creator: "autogpt",
              runs: 940,
            },
          ],
        },
      },
      {
        tool: "find_library_agent",
        input: { query: "market research" },
        output: {
          type: "agents_found",
          message: "Found 1 library agent",
          count: 1,
          agents: [
            {
              id: "lib-201",
              name: "EV Sales Tracker",
              description: "Monthly EV market pulls",
              source: "library",
              in_library: true,
            },
          ],
        },
      },
      {
        tool: "run_agent",
        input: { username_agent_slug: "autogpt/ev-sales-tracker" },
        output: {
          type: "execution_started",
          message: "Execution started",
          execution_id: "exec-91",
          graph_id: "graph-77",
          graph_name: "EV Sales Tracker",
          library_agent_id: "lib-201",
          library_agent_link: "/library/agents/lib-201",
          status: "COMPLETED",
        },
      },
      {
        tool: "view_agent_output",
        input: { execution_id: "exec-91" },
        output: {
          type: "agent_output",
          message: "Latest execution output",
          agent_name: "EV Sales Tracker",
          agent_id: "graph-77",
          library_agent_id: "lib-201",
          library_agent_link: "/library/agents/lib-201",
          total_executions: 3,
          execution: {
            execution_id: "exec-91",
            status: "COMPLETED",
            outputs: {
              report: ["17.1M units in 2024"],
              growth_yoy: ["+25%"],
            },
          },
        },
      },
      {
        tool: "run_sub_session",
        input: { prompt: "Research charging infrastructure growth in the EU" },
        output: {
          type: "mcp_tool_output",
          message: "Sub-AutoPilot finished",
          status: "completed",
          sub_session_id: "sub-11",
          response: "Found 3 key datapoints on EU charging growth…",
          sub_autopilot_session_id: "sess-sub-11",
          sub_autopilot_session_link: "/copilot?session=sess-sub-11",
          elapsed_seconds: 184.2,
        },
      },
      {
        tool: "get_sub_session_result",
        input: { sub_autopilot_session_id: "sess-sub-11" },
        output: {
          type: "mcp_tool_output",
          message: "Sub-AutoPilot still running",
          status: "running",
          sub_session_id: "sub-11",
          elapsed_seconds: 42.0,
        },
      },
    ],
  },
  {
    title: "Agent building",
    tools: [
      {
        tool: "decompose_goal",
        input: { goal: "Summarize Hacker News every morning" },
        output: {
          type: "task_decomposition",
          message: "Plan created",
          goal: "Summarize Hacker News every morning",
          step_count: 3,
          steps: [
            {
              step_id: "step_1",
              description: "Fetch the Hacker News front page",
              action: "add_block",
              block_name: "GetRequestBlock",
              status: "pending",
            },
            {
              step_id: "step_2",
              description: "Summarize the top stories",
              action: "add_block",
              block_name: "AITextGeneratorBlock",
              status: "pending",
            },
            {
              step_id: "step_3",
              description: "Email the digest",
              action: "add_block",
              block_name: "SendEmailBlock",
              status: "pending",
            },
          ],
        },
      },
      {
        tool: "create_agent",
        input: { save: true },
        output: {
          type: "agent_builder_saved",
          message: "Agent saved to your library",
          agent_id: "ag-1",
          agent_name: "HN Digest",
          graph_version: 1,
          library_agent_id: "lib-301",
          library_agent_link: "/library/agents/lib-301",
          agent_page_link: "/build?flowID=ag-1",
        },
      },
      {
        tool: "edit_agent",
        input: { agent_id: "ag-1" },
        output: {
          type: "agent_builder_saved",
          message: "Agent updated",
          agent_id: "ag-1",
          agent_name: "HN Digest",
          graph_version: 4,
          library_agent_id: "lib-301",
          library_agent_link: "/library/agents/lib-301",
          agent_page_link: "/build?flowID=ag-1",
        },
      },
      {
        tool: "customize_agent",
        input: { agent_json_ref: "@@agptfile:draft.json" },
        output: {
          type: "agent_builder_saved",
          message: "Customized copy saved",
          agent_id: "ag-2",
          agent_name: "HN Digest (mine)",
          library_agent_id: "lib-302",
          library_agent_link: "/library/agents/lib-302",
          agent_page_link: "/build?flowID=ag-2",
        },
      },
      {
        tool: "validate_agent_graph",
        input: { node_count: 6 },
        output: {
          type: "agent_builder_validation_result",
          message: "Graph is valid",
          valid: true,
          errors: [],
          error_count: 0,
        },
      },
      {
        tool: "fix_agent_graph",
        input: { errors: ["missing input link on node 4"] },
        output: {
          type: "agent_builder_fix_result",
          message: "Applied 1 fix",
          fixes_applied: ["linked node 4 input to node 3 output"],
          fix_count: 1,
          valid_after_fix: true,
          remaining_errors: [],
        },
      },
      {
        tool: "enter_agent_building_mode",
        output: { message: "Entered agent building mode" },
      },
    ],
  },
  {
    title: "Blocks",
    tools: [
      {
        tool: "find_block",
        input: { query: "send email" },
        output: {
          type: "block_list",
          message: "Found 2 blocks for 'send email'",
          count: 2,
          query: "send email",
          blocks: [
            {
              id: "b-9",
              name: "Send Gmail",
              description: "Send email via Gmail",
              categories: ["COMMUNICATION"],
              provider: "google",
            },
            {
              id: "b-10",
              name: "SMTP Send",
              description: "Send via any SMTP server",
              categories: ["COMMUNICATION"],
              provider: "smtp",
            },
          ],
        },
      },
      {
        tool: "run_block",
        input: { block_id: "b-9", input_data: { to: "team@agpt.co" } },
        output: {
          type: "block_output",
          message: "Block 'Send Gmail' executed successfully",
          block_id: "b-9",
          block_name: "Send Gmail",
          provider: "google",
          outputs: { status: ["sent"], message_id: ["m-1187"] },
          success: true,
        },
      },
      {
        tool: "continue_run_block",
        input: { block_id: "b-9" },
        output: {
          type: "block_output",
          message: "Block 'Send Gmail' executed successfully",
          block_id: "b-9",
          block_name: "Send Gmail",
          provider: "google",
          outputs: { status: ["sent"] },
          success: true,
        },
      },
    ],
  },
  {
    title: "Web & browser",
    tools: [
      {
        tool: "web_search",
        input: { query: "global EV sales 2024 total units" },
        output: {
          type: "web_search",
          message: "Search complete",
          query: "global EV sales 2024 total units",
          answer:
            "Global EV sales reached about 17.1M units in 2024, up ~25% year-over-year.",
          results: [
            {
              title: "EV sales hit 17.1M in 2024",
              url: "https://rhomotion.com/ev-sales-2024",
              snippet: "Rho Motion's final tally for 2024 EV sales…",
              page_age: "2025-01-14",
            },
            {
              title: "Global EV Outlook 2024",
              url: "https://www.iea.org/reports/global-ev-outlook-2024",
              snippet: "The IEA's annual outlook on electric mobility…",
            },
          ],
          search_requests: 1,
        },
      },
      {
        tool: "web_fetch",
        input: { url: "https://www.iea.org/reports/global-ev-outlook-2024" },
        output: {
          type: "web_fetch",
          message: "Fetched https://www.iea.org/reports/global-ev-outlook-2024",
          url: "https://www.iea.org/reports/global-ev-outlook-2024",
          status_code: 200,
          content_type: "text/html",
          title: "Global EV Outlook 2024 – Analysis - IEA",
          content: "# Global EV Outlook 2024\n\nElectric car sales neared…",
          content_length: 48213,
          truncated: false,
        },
      },
      {
        tool: "browser_navigate",
        input: { url: "https://news.ycombinator.com" },
        output: {
          type: "browser_navigate",
          message: "Navigated",
          url: "https://news.ycombinator.com",
          title: "Hacker News",
          snapshot: "- link 'More' @ref=e42",
        },
      },
      {
        tool: "browser_act",
        input: { action: "click", target: "More link" },
        output: {
          type: "browser_act",
          message: "Clicked",
          action: "click",
          current_url: "https://news.ycombinator.com/news?p=2",
          snapshot: "- link 'More' @ref=e42",
        },
      },
      {
        tool: "browser_screenshot",
        output: {
          type: "browser_screenshot",
          message: "Screenshot saved",
          file_id: "shot-1",
          filename: "screenshot-1.png",
        },
      },
    ],
  },
  {
    title: "Code & files",
    tools: [
      {
        tool: "bash_exec",
        input: { command: "python compute_ev_table.py --years 2023,2024" },
        output: {
          type: "bash_exec",
          message: "Command finished",
          stdout: "BYD  3.02M → 4.27M  (+41%)",
          stderr: "",
          exit_code: 0,
          timed_out: false,
        },
      },
      {
        tool: "read_workspace_file",
        input: { path: "ev-brief.md" },
        output: {
          type: "workspace_file_metadata",
          message: "Read ev-brief.md",
          file_id: "f-77",
          name: "ev-brief.md",
          path: "ev-brief.md",
          mime_type: "text/markdown",
          size_bytes: 2048,
          download_url: "workspace://f-77#text/markdown",
          preview: "# EV Brief\n\n17.1M units sold in 2024…",
        },
      },
      {
        tool: "write_workspace_file",
        input: { filename: "ev-brief.md" },
        output: {
          type: "workspace_file_written",
          message: "Wrote ev-brief.md",
          file_id: "f-77",
          name: "ev-brief.md",
          path: "ev-brief.md",
          mime_type: "text/markdown",
          size_bytes: 2048,
          download_url: "workspace://f-77#text/markdown",
          content_preview: "# EV Brief…",
        },
      },
      {
        tool: "delete_workspace_file",
        input: { path: "old-draft.md" },
        output: {
          type: "workspace_file_deleted",
          message: "Deleted old-draft.md",
          file_id: "f-12",
          success: true,
        },
      },
      {
        tool: "list_workspace_files",
        input: { path_prefix: "/" },
        output: {
          type: "workspace_file_list",
          message: "2 files",
          total_count: 2,
          files: [
            {
              file_id: "f-77",
              name: "ev-brief.md",
              path: "ev-brief.md",
              mime_type: "text/markdown",
              size_bytes: 2048,
            },
            {
              file_id: "f-78",
              name: "ev-growth.png",
              path: "ev-growth.png",
              mime_type: "image/png",
              size_bytes: 91433,
            },
          ],
        },
      },
      {
        tool: "Read",
        input: { file_path: "src/auth/middleware.ts" },
        output: "export function verifyToken(token: string) { … }",
      },
      {
        tool: "Write",
        input: { file_path: "src/auth/middleware.test.ts" },
        output: { created: true },
      },
      {
        tool: "Edit",
        input: { file_path: "src/auth/middleware.ts" },
        output: DIFF_SAMPLE,
      },
      {
        tool: "edit_file",
        input: { file_path: "src/auth/config.ts" },
        output: DIFF_SAMPLE,
      },
      {
        tool: "Glob",
        input: { pattern: "**/*.test.ts" },
        output: { count: 42 },
      },
      {
        tool: "Grep",
        input: { pattern: "jwt.verify" },
        output: { matches: 3 },
      },
    ],
  },
  {
    title: "Memory",
    tools: [
      {
        tool: "memory_search",
        input: { query: "user timezone" },
        output: {
          type: "memory_search",
          message: "1 fact found",
          facts: ["User is in IST"],
          recent_episodes: [],
        },
      },
      {
        tool: "memory_store",
        input: { name: "EV brief preferences" },
        output: {
          type: "memory_store",
          message: "Memory stored",
          memory_name: "EV brief preferences",
        },
      },
      {
        tool: "memory_forget_search",
        input: { query: "old address" },
        output: {
          type: "memory_forget_candidates",
          message: "2 candidates",
          candidates: [
            { uuid: "m-1", fact: "User lived in Pune" },
            { uuid: "m-2", fact: "User's old office address" },
          ],
        },
      },
      {
        tool: "memory_forget_confirm",
        output: {
          type: "memory_forget_confirm",
          message: "Deleted 2 memories",
          deleted_uuids: ["m-1", "m-2"],
          failed_uuids: [],
        },
      },
      {
        tool: "add_understanding",
        input: { note: "User ships 6 PRs a day" },
        output: {
          type: "understanding_updated",
          message: "Understanding updated",
          updated_fields: ["work_style"],
        },
      },
    ],
  },
  {
    title: "Folders",
    tools: [
      {
        tool: "create_folder",
        input: { name: "Research agents" },
        output: {
          type: "folder_created",
          message: "Folder created",
          folder: { id: "fo-1", name: "Research agents", agent_count: 0 },
        },
      },
      {
        tool: "list_folders",
        output: {
          type: "folder_list",
          message: "2 folders",
          count: 2,
          folders: [
            { id: "fo-1", name: "Research agents", agent_count: 3 },
            { id: "fo-2", name: "Ops", agent_count: 1 },
          ],
        },
      },
      {
        tool: "update_folder",
        input: { folder_id: "fo-1", name: "Research" },
        output: {
          type: "folder_updated",
          message: "Folder updated",
          folder: { id: "fo-1", name: "Research", agent_count: 3 },
        },
      },
      {
        tool: "move_folder",
        input: { folder_id: "fo-2", target_parent_id: null },
        output: {
          type: "folder_moved",
          message: "Folder moved",
          folder: { id: "fo-2", name: "Ops", agent_count: 1 },
        },
      },
      {
        tool: "delete_folder",
        input: { folder_id: "fo-3" },
        output: {
          type: "folder_deleted",
          message: "Folder deleted",
          folder_id: "fo-3",
        },
      },
      {
        tool: "move_agents_to_folder",
        input: { agent_ids: ["ag-1", "ag-2"] },
        output: {
          type: "agents_moved_to_folder",
          message: "Moved 2 agents",
          agent_ids: ["ag-1", "ag-2"],
          agent_names: ["HN Digest", "HN Digest (mine)"],
          folder_id: "fo-1",
          count: 2,
        },
      },
    ],
  },
  {
    title: "Schedules, triggers & presets",
    tools: [
      {
        tool: "list_schedules",
        output: {
          type: "schedule_list",
          message: "2 schedules",
          schedules: [
            {
              schedule_id: "s-1",
              kind: "graph",
              name: "Weekly digest",
              timezone: "Asia/Kolkata",
              next_run_time: "2026-08-10T09:00:00+05:30",
              cron: "0 9 * * 1",
              graph_id: "graph-77",
            },
            {
              schedule_id: "s-2",
              kind: "copilot_turn",
              name: "Monthly EV sales update",
              timezone: "Asia/Kolkata",
              next_run_time: "2026-09-01T09:00:00+05:30",
              cron: "0 9 1 * *",
            },
          ],
        },
      },
      {
        tool: "delete_schedule",
        input: { schedule_id: "s-1" },
        output: {
          type: "schedule_deleted",
          message: "Schedule deleted",
          schedule_id: "s-1",
        },
      },
      {
        tool: "schedule_followup",
        input: { message: "Monthly EV sales update", cron: "0 9 1 * *" },
        output: {
          type: "schedule_created",
          message: "Follow-up scheduled",
          schedule_id: "s-2",
          next_run_time: "2026-09-01T09:00:00+05:30",
          is_recurring: true,
        },
      },
      { tool: "list_agent_triggers", output: { triggers: 1 } },
      {
        tool: "setup_agent_webhook_trigger",
        output: { ingress_url: "https://hooks.agpt.co/t/abc" },
      },
      { tool: "list_presets", output: { presets: 3 } },
      {
        tool: "update_preset",
        input: { name: "Weekly digest" },
        output: { ok: true },
      },
      { tool: "delete_preset", output: { deleted: true } },
    ],
  },
  {
    title: "Chat platforms",
    tools: [
      {
        tool: "post_to_chat_platform",
        input: { channel: "#ev-updates" },
        output: {
          type: "chat_platform_posted",
          message: "Posted to #ev-updates",
          platform: "discord",
          kind: "message",
          channel_id: "ch-9",
          ref_id: "m-1",
          url: "https://discord.com/channels/1/ch-9/m-1",
        },
      },
      {
        tool: "list_chat_platform_channels",
        output: {
          type: "chat_platform_channel_list",
          message: "2 channels",
          platform: "discord",
          count: 2,
          channels: [
            { id: "ch-1", name: "general", server_id: "srv-1" },
            { id: "ch-9", name: "ev-updates", server_id: "srv-1" },
          ],
        },
      },
    ],
  },
  {
    title: "Docs & skills",
    tools: [
      {
        tool: "search_docs",
        input: { query: "scheduled agent runs" },
        output: {
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
            {
              title: "Presets",
              path: "platform/presets.md",
              section: "Saved runs",
              snippet: "Save input presets for repeat runs…",
              score: 0.74,
            },
          ],
        },
      },
      {
        tool: "get_doc_page",
        input: { path: "platform/scheduling.md" },
        output: {
          type: "doc_page",
          message: "Loaded page",
          title: "Scheduling agents",
          path: "platform/scheduling.md",
          content: "# Scheduling agents\n\nUse cron expressions to…",
        },
      },
      {
        tool: "get_agent_building_guide",
        output: { message: "Guide loaded (9 sections)" },
      },
      {
        tool: "get_mcp_guide",
        output: { message: "Guide loaded (4 sections)" },
      },
      {
        tool: "store_skill",
        input: { name: "deploy-checklist" },
        output: {
          type: "skill_stored",
          message: "Skill saved",
          name: "deploy-checklist",
          description: "Pre-deploy verification steps",
          triggers: ["deploy", "release"],
        },
      },
      {
        tool: "read_skill",
        input: { name: "deploy-checklist" },
        output: {
          type: "skill_loaded",
          message: "Skill loaded",
          name: "deploy-checklist",
          description: "Pre-deploy verification steps",
          body: "1. Run tests\n2. Check CI\n3. Tag release",
          triggers: ["deploy"],
          is_default: false,
        },
      },
      {
        tool: "delete_skill",
        input: { name: "old-skill" },
        output: {
          type: "skill_deleted",
          message: "Skill deleted",
          name: "old-skill",
        },
      },
      {
        tool: "list_skills",
        output: {
          type: "skill_list",
          message: "1 skill",
          skills: [
            {
              name: "deploy-checklist",
              description: "Pre-deploy verification steps",
            },
          ],
        },
      },
    ],
  },
  {
    title: "Integrations & MCP",
    tools: [
      {
        tool: "connect_integration",
        input: { provider: "github" },
        output: {
          type: "setup_requirements",
          message: "Connect GitHub to continue",
          setup_info: {
            agent_id: "connect_github",
            agent_name: "GitHub",
            user_readiness: { has_all_credentials: false, ready_to_run: false },
          },
        },
      },
      {
        tool: "connect_integration",
        input: { provider: "discord" },
        output: {
          type: "setup_requirements",
          message: "Connect Discord to continue",
          setup_info: {
            agent_id: "connect_discord",
            agent_name: "Discord",
            user_readiness: { has_all_credentials: false, ready_to_run: false },
          },
        },
      },
      {
        tool: "connect_integration",
        input: { provider: "notion" },
        output: {
          type: "setup_requirements",
          message: "Connect Notion to continue",
          setup_info: {
            agent_id: "connect_notion",
            agent_name: "Notion",
            user_readiness: { has_all_credentials: false, ready_to_run: false },
          },
        },
      },
      {
        tool: "run_mcp_tool",
        input: {
          server_url: "https://mcp.linear.app",
          tool_name: "create_issue",
        },
        output: {
          type: "mcp_tool_output",
          message: "MCP tool executed",
          server_url: "https://mcp.linear.app",
          tool_name: "create_issue",
          result: {
            identifier: "OPEN-3211",
            url: "https://linear.app/i/OPEN-3211",
          },
          success: true,
        },
      },
    ],
  },
  {
    title: "Misc",
    tools: [
      {
        tool: "TodoWrite",
        input: {
          todos: [
            {
              content: "Fetch sources",
              status: "completed",
              activeForm: "Fetching sources",
            },
            {
              content: "Write brief",
              status: "in_progress",
              activeForm: "Writing the brief",
            },
          ],
        },
        output: {
          type: "todo_write",
          message: "Todos updated",
          todos: [
            {
              content: "Fetch sources",
              status: "completed",
              activeForm: "Fetching sources",
            },
            {
              content: "Write brief",
              status: "in_progress",
              activeForm: "Writing the brief",
            },
          ],
        },
      },
      { tool: "context_compaction", output: { summarized_messages: 34 } },
      {
        tool: "search_feature_requests",
        input: { query: "dark mode" },
        output: {
          type: "feature_request_search",
          message: "5 matches",
          count: 5,
          query: "dark mode",
          results: [
            { id: "i-1", identifier: "FR-101", title: "Dark mode for builder" },
            { id: "i-2", identifier: "FR-230", title: "Dark mode scheduling" },
          ],
        },
      },
      {
        tool: "create_feature_request",
        input: { title: "Bulk agent import" },
        output: {
          type: "feature_request_created",
          message: "Feature request filed",
          issue_id: "i-9",
          issue_identifier: "FR-812",
          issue_title: "Bulk agent import",
          issue_url: "https://linear.app/i/FR-812",
          is_new_issue: true,
          customer_name: "Acme",
        },
      },
      {
        tool: "get_platform_info",
        output: {
          type: "platform_info",
          message: "You're on the Pro plan with 4,210 credits.",
          topic: "billing",
          tier: "Pro",
          billing_url: "/settings/billing",
        },
      },
      {
        tool: "ask_question",
        input: {
          questions: [
            {
              question: "Which channel should the digest go to?",
              keyword: "channel",
            },
          ],
        },
        output: {
          type: "agent_builder_clarification_needed",
          message: "Waiting for your answer",
          questions: [
            {
              question: "Which channel should the digest go to?",
              keyword: "channel",
              example: "#general, #ev-updates",
            },
          ],
        },
      },
    ],
  },
];

const ASK_QUESTION_SAMPLE: SampleTool = {
  tool: "ask_question",
  input: {
    questions: [
      {
        question: "Which channel should the digest go to?",
        keyword: "channel",
      },
      { question: "What time should it post?", keyword: "time" },
    ],
  },
  output: {
    type: "agent_builder_clarification_needed",
    message: "I need a couple of details before setting this up.",
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
};

// Interactive tools stay OUTSIDE the chain (CUSTOM_TOOL_TYPES) and render
// their real components — credential pickers, input forms, answer chips.
export const INTERACTIVE_SAMPLES: { label: string; sample: SampleTool }[] = [
  { label: "ask_question — user answers inline", sample: ASK_QUESTION_SAMPLE },
  {
    label: "connect_integration — credentials picker",
    sample: {
      tool: "connect_integration",
      input: { provider: "discord" },
      output: {
        type: "setup_requirements",
        message: "Connect Discord to continue",
        setup_info: {
          agent_id: "connect_discord",
          agent_name: "Discord",
          requirements: { credentials: [], inputs: [], execution_modes: [] },
          user_readiness: {
            has_all_credentials: false,
            ready_to_run: false,
            missing_credentials: {
              discord_credentials: {
                provider: "discord",
                types: ["oauth2"],
                title: "Discord",
              },
            },
          },
        },
      },
    },
  },
  {
    label: "run_agent — setup requirements (creds + inputs)",
    sample: {
      tool: "run_agent",
      input: { username_agent_slug: "autogpt/ev-sales-tracker" },
      output: {
        type: "setup_requirements",
        message: "Setup needed before this agent can run",
        setup_info: {
          agent_id: "graph-77",
          agent_name: "EV Sales Tracker",
          requirements: {
            credentials: [],
            inputs: [
              {
                name: "region",
                type: "string",
                description: "Market region to track",
                required: true,
              },
              {
                name: "months",
                type: "integer",
                description: "How many months back",
                required: false,
                default: 12,
              },
            ],
            execution_modes: ["immediate", "scheduled"],
          },
          user_readiness: {
            has_all_credentials: false,
            ready_to_run: false,
            missing_credentials: {
              google_credentials: {
                provider: "google",
                types: ["oauth2"],
                title: "Google",
                scopes: ["https://www.googleapis.com/auth/spreadsheets"],
              },
            },
          },
        },
      },
    },
  },
  {
    label: "run_block — human review required",
    sample: {
      tool: "run_block",
      input: { block_id: "b-9", input_data: { to: "team@agpt.co" } },
      output: {
        type: "review_required",
        message: "This action needs your approval before it runs.",
        block_id: "b-9",
        block_name: "Send Gmail",
        review_id: "rev-1",
        graph_exec_id: "copilot-session-demo",
        input_data: {
          to: "team@agpt.co",
          subject: "EV brief",
          body: "17.1M units in 2024…",
        },
      },
    },
  },
];

// A chain interrupted by a question: the chain splits, the interactive card
// renders full-width between the two chain halves (same behavior as
// ChatMessagesContainer's isChainableToolPart segmentation).
export const INTERRUPT_DEMO: SampleTool[] = [
  {
    reasoning:
      "Planning the digest agent. I know the data source, but the target channel and posting time are the user's call — asking before wiring the blocks.",
  },
  {
    tool: "web_search",
    input: { query: "discord digest bot best practices" },
    output: {
      type: "web_search",
      message: "Search complete",
      query: "discord digest bot best practices",
      results: [
        {
          title: "Designing daily digest bots",
          url: "https://discord.com/developers/docs",
        },
      ],
      search_requests: 1,
    },
  },
  ASK_QUESTION_SAMPLE,
  {
    tool: "bash_exec",
    input: { command: "python render_digest_preview.py" },
    output: {
      type: "bash_exec",
      message: "Command finished",
      stdout: "preview written to digest-preview.md",
      stderr: "",
      exit_code: 0,
      timed_out: false,
    },
  },
  {
    tool: "write_workspace_file",
    input: { filename: "digest-preview.md" },
    output: {
      type: "workspace_file_written",
      message: "Wrote digest-preview.md",
      file_id: "f-91",
      name: "digest-preview.md",
      path: "digest-preview.md",
      mime_type: "text/markdown",
      size_bytes: 1204,
      download_url: "workspace://f-91#text/markdown",
    },
  },
];

export const STATE_SAMPLES: SampleTool[] = [
  { tool: "web_search", state: "input-streaming" },
  {
    tool: "web_search",
    state: "input-available",
    input: { query: "global EV sales 2024" },
  },
  {
    tool: "web_search",
    state: "output-available",
    input: { query: "global EV sales 2024" },
    output: {
      type: "web_search",
      message: "Search complete",
      query: "global EV sales 2024",
      results: [
        {
          title: "EV sales hit 17.1M in 2024",
          url: "https://rhomotion.com/ev-sales-2024",
          snippet: "Final 2024 tally…",
        },
      ],
      search_requests: 1,
    },
  },
  {
    tool: "web_fetch",
    state: "output-error",
    input: { url: "https://example.com/missing" },
    errorText: "404 Not Found — page does not exist",
  },
];

const REASONING_TEXT =
  "Reading the request and locating the jwt.verify call inside the auth middleware. The verify call sets no algorithms allowlist, so a token signed with 'none' could be accepted. Planning to pin the algorithm to HS256 and validate issuer and audience claims on every request, then scan the existing tests so the fix stays covered.";

export const THINKING_SAMPLES: SampleTool[] = [
  { reasoning: REASONING_TEXT, reasoningStreaming: true },
  { reasoning: REASONING_TEXT },
];

export const CHAIN_DEMOS: {
  title: string;
  streaming: boolean;
  tools: SampleTool[];
}[] = [
  {
    title: "Streaming (sliding window + live step)",
    streaming: true,
    tools: [
      { reasoning: REASONING_TEXT },
      {
        tool: "web_search",
        input: { query: "EV sales 2024" },
        output: {
          type: "web_search",
          message: "Search complete",
          query: "EV sales 2024",
          results: [
            {
              title: "EV sales hit 17.1M in 2024",
              url: "https://rhomotion.com/ev-sales-2024",
            },
          ],
          search_requests: 1,
        },
      },
      {
        tool: "bash_exec",
        input: { command: "python compute_ev_table.py" },
        output: {
          type: "bash_exec",
          message: "Command finished",
          stdout: "BYD  3.02M → 4.27M  (+41%)",
          stderr: "",
          exit_code: 0,
          timed_out: false,
        },
      },
      {
        tool: "write_workspace_file",
        state: "input-available",
        input: { filename: "ev-brief.md" },
      },
    ],
  },
  {
    title: "Settled (collapsed — click to expand)",
    streaming: false,
    tools: [
      { reasoning: REASONING_TEXT },
      {
        tool: "TodoWrite",
        input: {
          todos: [
            { content: "Plan", status: "completed", activeForm: "Planning" },
          ],
        },
        output: { type: "todo_write", message: "Todos updated" },
      },
      {
        tool: "web_search",
        input: { query: "EV sales 2024" },
        output: {
          type: "web_search",
          message: "Search complete",
          query: "EV sales 2024",
          results: [
            {
              title: "EV sales hit 17.1M in 2024",
              url: "https://rhomotion.com/ev-sales-2024",
            },
          ],
          search_requests: 1,
        },
      },
      {
        tool: "Edit",
        input: { file_path: "src/auth/middleware.ts" },
        output: DIFF_SAMPLE,
      },
    ],
  },
  {
    title: "With a failed step",
    streaming: false,
    tools: [
      {
        tool: "web_fetch",
        state: "output-error",
        input: { url: "https://example.com/missing" },
        errorText: "404 Not Found",
      },
      {
        tool: "web_fetch",
        input: { url: "https://www.iea.org/reports/global-ev-outlook-2024" },
        output: {
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
      },
    ],
  },
];
