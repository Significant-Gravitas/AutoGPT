import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { ChainRow } from "../copilot/components/ToolChain/helpers";
import { getCatalogLabel } from "../copilot/components/ToolChain/toolCatalog";

interface Sample {
  title: string;
  label: string;
  row: ChainRow;
}

function sample(
  title: string,
  tool: string,
  input: Record<string, unknown>,
  output: Record<string, unknown>,
): Sample {
  const catalog = getCatalogLabel(tool, input, "done");
  return {
    title,
    label: catalog?.text ?? tool,
    row: {
      key: tool,
      category: catalog?.category ?? "other",
      text: catalog?.text ?? tool,
      state: "done",
      tool,
      input,
      output,
    },
  };
}

export const SAMPLE_ROWS: Sample[] = [
  sample(
    "hire_expert · preview",
    "hire_expert",
    { template_id: "tpl-scout" },
    {
      type: "expert_change_proposed",
      message: "Nothing hired yet.",
      applied: false,
      confirmation_id: "c-1",
      preview: {
        kind: "hire",
        name: "Scout",
        role: "Market research",
        about:
          "You track competitors and summarise what changed each week, with sources.",
        boundaries: "You never contact anyone outside the team.",
        voice_preferences: "Short, factual, no hype.",
        avatar_url: null,
        color: "sky-300",
      },
    },
  ),
  sample(
    "raise_expert · preview",
    "raise_expert",
    { name: "Otto" },
    {
      type: "expert_change_proposed",
      message: "Nothing created yet.",
      applied: false,
      confirmation_id: "c-2",
      preview: {
        kind: "raise",
        name: "Otto",
        role: "Inbox triage",
        about:
          "You read the shared inbox each morning, group what arrived, and draft replies for anything routine.",
        boundaries:
          "You never send a reply yourself — everything goes to the user for a final look.",
        weekly_budget: 2000,
      },
    },
  ),
  sample(
    "confirm_expert_change · applied",
    "confirm_expert_change",
    { confirmation_id: "c-2" },
    {
      type: "expert_change_applied",
      message: "Otto is raised and on the team.",
      applied: true,
      kind: "raise",
      expert: {
        id: "exp-otto",
        name: "Otto",
        role: "Inbox triage",
        avatar_url: null,
        color: "violet",
      },
    },
  ),
  sample(
    "handoff_to_expert · queued",
    "handoff_to_expert",
    { expert_id: "exp-otto", prompt: "Take over the weekly inbox summary" },
    {
      type: "mcp_tool_output",
      message: "Otto picked the task up.",
      status: "running",
      sub_session_id: "sub-1",
      sub_autopilot_session_link: "/copilot?sessionId=sub-1",
      elapsed_seconds: 1,
      expert: {
        id: "exp-otto",
        name: "Otto",
        role: "Inbox triage",
        avatar_url: null,
        color: "violet",
      },
    },
  ),
];

export const SAMPLE_QUESTION_ITEM: HomeAttentionItem = {
  id: "question-sess-1",
  kind: "question",
  priority: "normal",
  title: "Scout has a question",
  description:
    "Do you want the weekly competitor summary on Monday morning or Friday evening?",
  why_it_matters: "The work is paused until you answer in the chat.",
  expert: {
    id: "exp-scout",
    name: "Scout",
    role: "Market research",
    avatar_url: null,
  },
  created_at: new Date("2026-08-21T08:00:00Z"),
  primary_action: { label: "Answer", href: "/copilot?sessionId=sess-1" },
};
