"use client";

import {
  Conversation,
  ConversationContent,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { CopilotChatActionsProvider } from "../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { FindBlocksTool } from "../tools/FindBlocks/FindBlocks";
import { FindAgentsTool } from "../tools/FindAgents/FindAgents";
import { SearchDocsTool } from "../tools/SearchDocs/SearchDocs";
import { RunBlockTool } from "../tools/RunBlock/RunBlock";
import { RunAgentTool } from "../tools/RunAgent/RunAgent";
import { CreateAgentTool } from "../tools/CreateAgent/CreateAgent";
import { EditAgentTool } from "../tools/EditAgent/EditAgent";
import { ViewAgentOutputTool } from "../tools/ViewAgentOutput/ViewAgentOutput";
import { ResponseType } from "@/app/api/__generated__/models/responseType";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-10">
      <h2 className="mb-4 border-b border-neutral-200 pb-2 text-lg font-semibold text-neutral-800">
        {title}
      </h2>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function SubSection({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-neutral-300 p-3">
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-neutral-500">
        {label}
      </p>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mock data factories
// ---------------------------------------------------------------------------

let _id = 0;
function uid() {
  return `sg-${++_id}`;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function StyleguidePage() {
  return (
    <CopilotChatActionsProvider onSend={(msg) => alert(`onSend: ${msg}`)}>
      <div className="h-[calc(100vh-72px)] overflow-y-auto bg-[#f8f8f9]">
        <div className="mx-auto max-w-3xl px-4 py-10">
          <h1 className="mb-1 text-2xl font-bold text-neutral-900">
            Copilot Styleguide
          </h1>
          <p className="mb-8 text-sm text-neutral-500">
            Static showcase of all chat message types, tool states &amp;
            variants.
          </p>

          {/* ============================================================= */}
          {/* MESSAGE TYPES                                                  */}
          {/* ============================================================= */}

          <Section title="Messages">
            <SubSection label="User message">
              <Message from="user">
                <MessageContent className="text-[1rem] leading-relaxed group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0]">
                  <MessageResponse>
                    Find me an agent that can summarize YouTube videos
                  </MessageResponse>
                </MessageContent>
              </Message>
            </SubSection>

            <SubSection label="Assistant message (text)">
              <Message from="assistant">
                <MessageContent className="text-[1rem] leading-relaxed group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900">
                  <MessageResponse>
                    I found a few agents that can help with YouTube video
                    summarization. Let me search for the best options for you.
                  </MessageResponse>
                </MessageContent>
              </Message>
            </SubSection>

            <SubSection label="Assistant message (markdown)">
              <Message from="assistant">
                <MessageContent className="text-[1rem] leading-relaxed group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900">
                  <MessageResponse>
                    {`Here's what I found:\n\n1. **YouTube Summarizer** — Extracts key points from any YouTube video\n2. **Video Digest** — Creates bullet-point summaries with timestamps\n\n> Both agents support videos up to 2 hours long.\n\n\`\`\`python\n# Example usage\nresult = agent.run(url="https://youtube.com/watch?v=...")\nprint(result.summary)\n\`\`\``}
                  </MessageResponse>
                </MessageContent>
              </Message>
            </SubSection>

            <SubSection label="Thinking state">
              <Message from="assistant">
                <MessageContent className="text-[1rem] leading-relaxed">
                  <span className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-transparent">
                    Thinking...
                  </span>
                </MessageContent>
              </Message>
            </SubSection>

            <SubSection label="Error state">
              <div className="rounded-lg bg-red-50 p-3 text-red-600">
                Error: Connection timed out. Please try again.
              </div>
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* FIND BLOCKS                                                    */}
          {/* ============================================================= */}

          <Section title="Tool: Find Blocks">
            <SubSection label="Input streaming">
              <FindBlocksTool
                part={{
                  type: "tool-find_block",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { query: "weather" },
                }}
              />
            </SubSection>

            <SubSection label="Input available">
              <FindBlocksTool
                part={{
                  type: "tool-find_block",
                  toolCallId: uid(),
                  state: "input-available",
                  input: { query: "weather" },
                }}
              />
            </SubSection>

            <SubSection label="Output available (with results)">
              <FindBlocksTool
                part={{
                  type: "tool-find_block",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { query: "weather" },
                  output: {
                    type: ResponseType.block_list,
                    blocks: [
                      {
                        id: "block-1",
                        name: "Get Weather",
                        description:
                          "Fetches current weather data for a given city using the OpenWeatherMap API.",
                        categories: [],
                      },
                      {
                        id: "block-2",
                        name: "Weather Forecast",
                        description:
                          "Returns a 5-day weather forecast for any location worldwide.",
                        categories: [],
                      },
                      {
                        id: "block-3",
                        name: "Historical Weather",
                        description:
                          "Retrieves historical weather data for the past 30 days.",
                        categories: [],
                      },
                    ],
                    count: 3,
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <FindBlocksTool
                part={{
                  type: "tool-find_block",
                  toolCallId: uid(),
                  state: "output-error",
                  input: { query: "weather" },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* FIND AGENTS (Marketplace)                                      */}
          {/* ============================================================= */}

          <Section title="Tool: Find Agents (Marketplace)">
            <SubSection label="Input streaming">
              <FindAgentsTool
                part={{
                  type: "tool-find_agent",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { query: "youtube summarizer" },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agents found)">
              <FindAgentsTool
                part={{
                  type: "tool-find_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { query: "youtube summarizer" },
                  output: {
                    type: ResponseType.agents_found,
                    agents: [
                      {
                        id: "creator/youtube-summarizer",
                        name: "YouTube Summarizer",
                        description:
                          "Summarizes YouTube videos into concise bullet points with key insights.",
                        source: "marketplace",
                      },
                      {
                        id: "creator/video-digest",
                        name: "Video Digest Pro",
                        description:
                          "Creates detailed summaries with timestamps from any YouTube video.",
                        source: "marketplace",
                      },
                    ],
                    count: 2,
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (no results)">
              <FindAgentsTool
                part={{
                  type: "tool-find_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { query: "quantum computing simulator" },
                  output: {
                    type: ResponseType.no_results,
                    message: "No agents found matching your query.",
                    suggestions: [
                      "Try broader search terms",
                      "Check the marketplace for related agents",
                    ],
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <FindAgentsTool
                part={{
                  type: "tool-find_agent",
                  toolCallId: uid(),
                  state: "output-error",
                  input: { query: "youtube summarizer" },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* FIND AGENTS (Library)                                          */}
          {/* ============================================================= */}

          <Section title="Tool: Find Agents (Library)">
            <SubSection label="Input streaming">
              <FindAgentsTool
                part={{
                  type: "tool-find_library_agent",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { query: "social media" },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agents found)">
              <FindAgentsTool
                part={{
                  type: "tool-find_library_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { query: "social media" },
                  output: {
                    type: ResponseType.agents_found,
                    agents: [
                      {
                        id: "lib-agent-1",
                        name: "Twitter Post Scheduler",
                        description:
                          "Schedules and posts tweets at optimal times for engagement.",
                        source: "library",
                      },
                    ],
                    count: 1,
                  },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* SEARCH DOCS                                                    */}
          {/* ============================================================= */}

          <Section title="Tool: Search Docs">
            <SubSection label="Input streaming">
              <SearchDocsTool
                part={{
                  type: "tool-search_docs",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { query: "webhooks" },
                }}
              />
            </SubSection>

            <SubSection label="Output available (search results)">
              <SearchDocsTool
                part={{
                  type: "tool-search_docs",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { query: "webhooks" },
                  output: {
                    type: ResponseType.doc_search_results,
                    query: "webhooks",
                    count: 2,
                    results: [
                      {
                        title: "Webhook Configuration Guide",
                        path: "platform/webhooks/configuration.md",
                        section: "Setup",
                        snippet:
                          "Webhooks allow you to receive real-time notifications when events occur in your AutoGPT workspace. Configure webhook endpoints to integrate with external services.",
                        doc_url:
                          "https://docs.agpt.co/platform/webhooks/configuration",
                      },
                      {
                        title: "Webhook Events Reference",
                        path: "platform/webhooks/events.md",
                        section: "Event Types",
                        snippet:
                          "A complete reference of all webhook event types including agent.completed, execution.started, and execution.failed events.",
                        doc_url:
                          "https://docs.agpt.co/platform/webhooks/events",
                      },
                    ],
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (no results)">
              <SearchDocsTool
                part={{
                  type: "tool-search_docs",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { query: "foobar" },
                  output: {
                    type: ResponseType.no_results,
                    message: "No documentation found for this query.",
                    suggestions: [
                      "Try different keywords",
                      "Check the documentation index",
                    ],
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <SearchDocsTool
                part={{
                  type: "tool-search_docs",
                  toolCallId: uid(),
                  state: "output-error",
                  input: { query: "webhooks" },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* GET DOC PAGE                                                   */}
          {/* ============================================================= */}

          <Section title="Tool: Get Doc Page">
            <SubSection label="Input streaming">
              <SearchDocsTool
                part={{
                  type: "tool-get_doc_page",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { path: "platform/getting-started.md" },
                }}
              />
            </SubSection>

            <SubSection label="Output available (page loaded)">
              <SearchDocsTool
                part={{
                  type: "tool-get_doc_page",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { path: "platform/getting-started.md" },
                  output: {
                    type: ResponseType.doc_page,
                    title: "Getting Started with AutoGPT Platform",
                    path: "platform/getting-started.md",
                    content:
                      "Welcome to AutoGPT Platform! This guide will walk you through setting up your first agent.\n\n## Prerequisites\n- An AutoGPT account\n- Basic understanding of automation workflows\n\n## Quick Start\n1. Navigate to the Builder\n2. Create a new agent\n3. Add blocks to your workflow\n4. Test and deploy",
                    doc_url: "https://docs.agpt.co/platform/getting-started",
                  },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* RUN BLOCK                                                      */}
          {/* ============================================================= */}

          <Section title="Tool: Run Block">
            <SubSection label="Input streaming">
              <RunBlockTool
                part={{
                  type: "tool-run_block",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { block_id: "weather-block-123" },
                }}
              />
            </SubSection>

            <SubSection label="Input available">
              <RunBlockTool
                part={{
                  type: "tool-run_block",
                  toolCallId: uid(),
                  state: "input-available",
                  input: {
                    block_id: "weather-block-123",
                    input_data: { city: "San Francisco" },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (block output)">
              <RunBlockTool
                part={{
                  type: "tool-run_block",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { block_id: "weather-block-123" },
                  output: {
                    type: ResponseType.block_output,
                    block_id: "weather-block-123",
                    block_name: "Get Weather",
                    message:
                      "Successfully fetched weather data for San Francisco.",
                    outputs: {
                      temperature: ["72°F"],
                      condition: ["Partly Cloudy"],
                      humidity: ["65%"],
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (setup requirements)">
              <RunBlockTool
                part={{
                  type: "tool-run_block",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { block_id: "weather-block-123" },
                  output: {
                    type: ResponseType.setup_requirements,
                    message:
                      "This block requires API credentials to run. Please configure them below.",
                    setup_info: {
                      agent_name: "Weather Agent",
                      requirements: {
                        inputs: [
                          {
                            name: "city",
                            title: "City",
                            type: "string",
                            required: true,
                            description: "The city to get weather for",
                          },
                        ],
                      },
                      user_readiness: {
                        missing_credentials: {
                          openweathermap: {
                            provider: "openweathermap",
                            credentials_type: "api_key",
                            title: "OpenWeatherMap API Key",
                            description:
                              "Required to access weather data. Get your key at openweathermap.org",
                          },
                        },
                      },
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (error)">
              <RunBlockTool
                part={{
                  type: "tool-run_block",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { block_id: "weather-block-123" },
                  output: {
                    type: ResponseType.error,
                    message: "Failed to run the block.",
                    error: "Block execution timed out after 30 seconds.",
                    details: {
                      block_id: "weather-block-123",
                      timeout_ms: 30000,
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <RunBlockTool
                part={{
                  type: "tool-run_block",
                  toolCallId: uid(),
                  state: "output-error",
                  input: { block_id: "weather-block-123" },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* RUN AGENT                                                      */}
          {/* ============================================================= */}

          <Section title="Tool: Run Agent">
            <SubSection label="Input streaming">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: { username_agent_slug: "creator/my-agent" },
                }}
              />
            </SubSection>

            <SubSection label="Output available (execution started)">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { username_agent_slug: "creator/my-agent" },
                  output: {
                    type: ResponseType.execution_started,
                    execution_id: "exec-abc-123-def-456",
                    graph_name: "YouTube Summarizer",
                    message:
                      "Agent execution started. You can track progress in your library.",
                    status: "running",
                    library_agent_link: "/library/agents/lib-123",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agent details / inputs needed)">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { username_agent_slug: "creator/my-agent" },
                  output: {
                    type: ResponseType.agent_details,
                    agent: {
                      name: "YouTube Summarizer",
                      description: "Summarizes YouTube videos into key points.",
                      inputs: [
                        {
                          name: "video_url",
                          title: "Video URL",
                          type: "string",
                          required: true,
                          description: "The YouTube video URL to summarize",
                        },
                        {
                          name: "language",
                          title: "Output Language",
                          type: "string",
                          required: false,
                          description:
                            "Language for the summary (default: English)",
                        },
                      ],
                    },
                    message: "This agent requires inputs to run.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (setup requirements)">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { username_agent_slug: "creator/my-agent" },
                  output: {
                    type: ResponseType.setup_requirements,
                    message: "This agent requires additional setup.",
                    setup_info: {
                      agent_name: "YouTube Summarizer",
                      requirements: {},
                      user_readiness: {
                        missing_credentials: {
                          youtube_api: {
                            provider: "youtube",
                            credentials_type: "api_key",
                            title: "YouTube Data API Key",
                            description:
                              "Required to access YouTube video data.",
                          },
                        },
                      },
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (need login)">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { username_agent_slug: "creator/my-agent" },
                  output: {
                    type: ResponseType.need_login,
                    message:
                      "You need to sign in to run this agent. Please log in and try again.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (error)">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { username_agent_slug: "creator/my-agent" },
                  output: {
                    type: ResponseType.error,
                    message: "Failed to start the agent.",
                    error: "Agent not found or has been deleted.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <RunAgentTool
                part={{
                  type: "tool-run_agent",
                  toolCallId: uid(),
                  state: "output-error",
                  input: { username_agent_slug: "creator/my-agent" },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* SCHEDULE AGENT                                                 */}
          {/* ============================================================= */}

          <Section title="Tool: Schedule Agent">
            <SubSection label="Input streaming (schedule)">
              <RunAgentTool
                part={{
                  type: "tool-schedule_agent",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: {
                    username_agent_slug: "creator/daily-digest",
                    schedule_name: "Morning Digest",
                    cron: "0 8 * * *",
                    timezone: "America/Los_Angeles",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (execution started — scheduled)">
              <RunAgentTool
                part={{
                  type: "tool-schedule_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  input: {
                    username_agent_slug: "creator/daily-digest",
                    schedule_name: "Morning Digest",
                    cron: "0 8 * * *",
                  },
                  output: {
                    type: ResponseType.execution_started,
                    execution_id: "sched-abc-123",
                    graph_name: "Daily Digest",
                    message:
                      "Agent scheduled successfully. It will run daily at 8:00 AM.",
                    status: "scheduled",
                    library_agent_link: "/library/agents/lib-456",
                  },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* CREATE AGENT                                                   */}
          {/* ============================================================= */}

          <Section title="Tool: Create Agent">
            <SubSection label="Input streaming">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "input-streaming",
                }}
              />
            </SubSection>

            <SubSection label="Output available (operation started)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.operation_started,
                    operation_id: "op-create-123",
                    tool_name: "create_agent",
                    message:
                      "Agent creation has been started. This may take a moment.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (operation pending)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.operation_pending,
                    operation_id: "op-create-123",
                    tool_name: "create_agent",
                    message:
                      "Agent creation is queued and will begin shortly.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (operation in progress)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.operation_in_progress,
                    tool_call_id: "tc-456",
                    message:
                      "An agent creation operation is already in progress. Please wait for it to finish.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agent preview)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.agent_preview,
                    agent_name: "Email Summarizer",
                    description:
                      "An agent that summarizes your unread emails into a daily digest.",
                    node_count: 4,
                    message:
                      "Here's a preview of the agent. Would you like me to save it?",
                    agent_json: {
                      name: "Email Summarizer",
                      nodes: [
                        { id: "1", type: "gmail_reader" },
                        { id: "2", type: "text_summarizer" },
                        { id: "3", type: "formatter" },
                        { id: "4", type: "output" },
                      ],
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agent saved)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.agent_saved,
                    agent_id: "agent-789",
                    agent_name: "Email Summarizer",
                    library_agent_id: "lib-agent-789",
                    library_agent_link: "/library/agents/lib-agent-789",
                    agent_page_link: "/build?agent=agent-789",
                    message:
                      "Agent 'Email Summarizer' has been saved to your library!",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (clarification needed)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.clarification_needed,
                    message:
                      "I need a bit more information before creating this agent.",
                    questions: [
                      {
                        question:
                          "Which email provider should the agent connect to?",
                        keyword: "email_provider",
                        example: "Gmail, Outlook, Yahoo",
                      },
                      {
                        question:
                          "How frequently should the digest be generated?",
                        keyword: "frequency",
                        example: "Daily, Twice a day, Weekly",
                      },
                    ],
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (error)">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.error,
                    message: "Failed to create the agent.",
                    error:
                      "The requested blocks are not compatible with each other.",
                    details: {
                      incompatible: ["gmail_reader", "slack_writer"],
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <CreateAgentTool
                part={{
                  type: "tool-create_agent",
                  toolCallId: uid(),
                  state: "output-error",
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* EDIT AGENT                                                     */}
          {/* ============================================================= */}

          <Section title="Tool: Edit Agent">
            <SubSection label="Input streaming">
              <EditAgentTool
                part={{
                  type: "tool-edit_agent",
                  toolCallId: uid(),
                  state: "input-streaming",
                }}
              />
            </SubSection>

            <SubSection label="Output available (operation started)">
              <EditAgentTool
                part={{
                  type: "tool-edit_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.operation_started,
                    operation_id: "op-edit-456",
                    tool_name: "edit_agent",
                    message: "Agent editing has started.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agent preview)">
              <EditAgentTool
                part={{
                  type: "tool-edit_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.agent_preview,
                    agent_name: "Email Summarizer v2",
                    description:
                      "Updated agent with improved summarization and Slack integration.",
                    node_count: 6,
                    message:
                      "Here's the updated agent. Shall I save these changes?",
                    agent_json: {
                      name: "Email Summarizer v2",
                      nodes: [
                        { id: "1", type: "gmail_reader" },
                        { id: "2", type: "text_summarizer" },
                        { id: "3", type: "formatter" },
                        { id: "4", type: "slack_writer" },
                        { id: "5", type: "scheduler" },
                        { id: "6", type: "output" },
                      ],
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (agent saved)">
              <EditAgentTool
                part={{
                  type: "tool-edit_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.agent_saved,
                    agent_id: "agent-789",
                    agent_name: "Email Summarizer v2",
                    library_agent_id: "lib-agent-789",
                    library_agent_link: "/library/agents/lib-agent-789",
                    agent_page_link: "/build?agent=agent-789",
                    message:
                      "Agent 'Email Summarizer v2' has been saved successfully!",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (clarification needed)">
              <EditAgentTool
                part={{
                  type: "tool-edit_agent",
                  toolCallId: uid(),
                  state: "output-available",
                  output: {
                    type: ResponseType.clarification_needed,
                    message:
                      "I need to clarify a few things about the edits you want.",
                    questions: [
                      {
                        question:
                          "Should the Slack notification replace the email output or be in addition to it?",
                        keyword: "notification_mode",
                        example: "Replace, Add alongside",
                      },
                    ],
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <EditAgentTool
                part={{
                  type: "tool-edit_agent",
                  toolCallId: uid(),
                  state: "output-error",
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* VIEW AGENT OUTPUT                                              */}
          {/* ============================================================= */}

          <Section title="Tool: View Agent Output">
            <SubSection label="Input streaming">
              <ViewAgentOutputTool
                part={{
                  type: "tool-view_agent_output",
                  toolCallId: uid(),
                  state: "input-streaming",
                  input: {
                    agent_name: "YouTube Summarizer",
                    execution_id: "exec-abc-123",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (with execution data)">
              <ViewAgentOutputTool
                part={{
                  type: "tool-view_agent_output",
                  toolCallId: uid(),
                  state: "output-available",
                  input: {
                    agent_name: "YouTube Summarizer",
                    execution_id: "exec-abc-123",
                  },
                  output: {
                    type: ResponseType.agent_output,
                    agent_id: "agent-123",
                    agent_name: "YouTube Summarizer",
                    message:
                      "Here are the results from the last execution.",
                    library_agent_link: "/library/agents/lib-agent-123",
                    execution: {
                      execution_id: "exec-abc-123",
                      status: "completed",
                      inputs_summary: {
                        video_url: "https://youtube.com/watch?v=dQw4w9WgXcQ",
                      },
                      outputs: {
                        summary: [
                          "Key Points:\n1. Introduction to the topic\n2. Main arguments presented\n3. Conclusion and takeaways",
                        ],
                        keywords: ["music", "classic", "80s"],
                      },
                    },
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (no execution selected)">
              <ViewAgentOutputTool
                part={{
                  type: "tool-view_agent_output",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { agent_name: "YouTube Summarizer" },
                  output: {
                    type: ResponseType.agent_output,
                    agent_id: "agent-123",
                    agent_name: "YouTube Summarizer",
                    message: "Agent found but no specific execution selected.",
                    library_agent_link: "/library/agents/lib-agent-123",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (no results)">
              <ViewAgentOutputTool
                part={{
                  type: "tool-view_agent_output",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { agent_name: "Nonexistent Agent" },
                  output: {
                    type: ResponseType.no_results,
                    message:
                      "No outputs found for this agent. It may not have been run yet.",
                    suggestions: [
                      "Try running the agent first",
                      "Check the agent name is correct",
                    ],
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output available (error)">
              <ViewAgentOutputTool
                part={{
                  type: "tool-view_agent_output",
                  toolCallId: uid(),
                  state: "output-available",
                  input: { agent_name: "YouTube Summarizer" },
                  output: {
                    type: ResponseType.error,
                    message: "Failed to retrieve agent output.",
                    error: "Execution data is corrupted or unavailable.",
                  },
                }}
              />
            </SubSection>

            <SubSection label="Output error">
              <ViewAgentOutputTool
                part={{
                  type: "tool-view_agent_output",
                  toolCallId: uid(),
                  state: "output-error",
                  input: { agent_name: "YouTube Summarizer" },
                }}
              />
            </SubSection>
          </Section>

          {/* ============================================================= */}
          {/* FULL CONVERSATION EXAMPLE                                      */}
          {/* ============================================================= */}

          <Section title="Full Conversation Example">
            <Conversation className="min-h-0 rounded-lg border bg-white">
              <ConversationContent className="gap-6 px-3 py-6">
                <Message from="user">
                  <MessageContent className="text-[1rem] leading-relaxed group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0]">
                    <MessageResponse>
                      Find me a block that can fetch weather data
                    </MessageResponse>
                  </MessageContent>
                </Message>

                <Message from="assistant">
                  <MessageContent className="text-[1rem] leading-relaxed group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900">
                    <MessageResponse>
                      Let me search for weather-related blocks for you.
                    </MessageResponse>

                    <FindBlocksTool
                      part={{
                        type: "tool-find_block",
                        toolCallId: uid(),
                        state: "output-available",
                        input: { query: "weather" },
                        output: {
                          type: ResponseType.block_list,
                          blocks: [
                            {
                              id: "block-1",
                              name: "Get Weather",
                              description:
                                "Fetches current weather data for a given city.",
                              categories: [],
                            },
                            {
                              id: "block-2",
                              name: "Weather Forecast",
                              description:
                                "Returns a 5-day weather forecast for any location.",
                              categories: [],
                            },
                          ],
                          count: 2,
                        },
                      }}
                    />

                    <MessageResponse>
                      I found 2 blocks related to weather. The **Get Weather**
                      block fetches current conditions, while **Weather
                      Forecast** provides a 5-day outlook. Would you like me to
                      run one of these?
                    </MessageResponse>
                  </MessageContent>
                </Message>

                <Message from="user">
                  <MessageContent className="text-[1rem] leading-relaxed group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0]">
                    <MessageResponse>
                      Yes, run the Get Weather block for San Francisco
                    </MessageResponse>
                  </MessageContent>
                </Message>

                <Message from="assistant">
                  <MessageContent className="text-[1rem] leading-relaxed group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900">
                    <RunBlockTool
                      part={{
                        type: "tool-run_block",
                        toolCallId: uid(),
                        state: "output-available",
                        input: {
                          block_id: "block-1",
                          input_data: { city: "San Francisco" },
                        },
                        output: {
                          type: ResponseType.block_output,
                          block_id: "block-1",
                          block_name: "Get Weather",
                          message:
                            "Successfully fetched weather for San Francisco.",
                          outputs: {
                            temperature: ["68°F"],
                            condition: ["Foggy"],
                            humidity: ["85%"],
                            wind: ["12 mph W"],
                          },
                        },
                      }}
                    />

                    <MessageResponse>
                      The current weather in San Francisco is **68°F** and
                      **Foggy** with 85% humidity and winds from the west at 12
                      mph.
                    </MessageResponse>
                  </MessageContent>
                </Message>
              </ConversationContent>
            </Conversation>
          </Section>
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
