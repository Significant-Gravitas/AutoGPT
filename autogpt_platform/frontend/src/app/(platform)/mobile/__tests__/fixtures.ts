import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";

export const expertFixture = {
  id: "maria",
  name: "Maria",
  role: "Marketing strategist",
  avatar_url: null,
  is_archived: false,
};
export const attentionFixture: HomeAttentionItem[] = [
  {
    id: "question-session-1",
    kind: "question",
    priority: "normal",
    title: "Maria has a question",
    description: "Which audience should the launch plan focus on?",
    why_it_matters: "The plan is waiting for your answer.",
    expert: expertFixture,
    primary_action: { label: "Answer", href: "/home?sessionId=session-1" },
  },
  {
    id: "approval-1",
    kind: "approval",
    priority: "high",
    title: "Review the launch email",
    description:
      "Otto is waiting for your review before the workflow continues.",
    why_it_matters: "Nothing is sent until you decide.",
    primary_action: { label: "Review", href: "/home?sessionId=session-2" },
    review: {
      node_exec_id: "node-1",
      graph_exec_id: "run-1",
      graph_id: "graph-1",
      graph_version: 1,
      user_id: "fixture-user",
      payload: {},
      editable: false,
      status: "WAITING",
      created_at: new Date("2026-10-07T12:00:00Z"),
    },
  },
];
export const chatsFixture = {
  sessions: [
    {
      id: "session-1",
      title: "Plan the autumn launch",
      expert_id: "maria",
      is_processing: false,
      created_at: "2026-10-07",
      updated_at: "2026-10-07",
    },
    {
      id: "session-2",
      title: "Review the launch email",
      expert_id: null,
      is_processing: false,
      created_at: "2026-10-07",
      updated_at: "2026-10-07",
    },
  ],
  total: 2,
};
