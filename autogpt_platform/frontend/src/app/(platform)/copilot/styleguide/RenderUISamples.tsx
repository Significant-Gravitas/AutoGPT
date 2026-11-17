import type { ToolUIPart } from "ai";
import { RenderUI } from "../tools/RenderUI/RenderUI";

const BLOCKS = [
  {
    type: "stats",
    items: [
      { label: "Leads", value: "412", change: "+6%", tone: "positive" },
      { label: "Replies", value: "96", change: "+37%", tone: "positive" },
      { label: "Meetings", value: "21", change: "+40%", tone: "positive" },
      { label: "Bounced", value: "31", change: "+4", tone: "negative" },
    ],
  },
  {
    type: "chart",
    title: "Replies per week",
    kind: "area",
    x_labels: ["Wk 1", "Wk 2", "Wk 3", "Wk 4"],
    series: [
      { name: "This month", values: [14, 17, 28, 37] },
      { name: "Last month", values: [15, 18, 17, 20] },
    ],
    annotation: { x: "Wk 3", label: "New subject lines" },
  },
  {
    type: "workflow_diagram",
    title: "Where leads drop out",
    steps: [
      { label: "New lead", detail: "412", status: "ok" },
      { label: "Enrich lead", detail: "31 failed", status: "error" },
      { label: "Draft email", detail: "381", status: "ok" },
      { label: "Send", detail: "381", status: "idle" },
    ],
  },
  {
    type: "timeline",
    title: "4-week product launch",
    items: [
      {
        date: "Week 1",
        title: "Prepare",
        detail: "Lock the offer, pricing and launch date.",
        status: "ok",
      },
      {
        date: "Week 2",
        title: "Build",
        detail: "Finish the landing page, launch emails and social posts.",
        status: "warning",
      },
      {
        date: "Week 3",
        title: "Warm up",
        detail: "Announce to your list and open early access.",
      },
      {
        date: "Week 4",
        title: "Launch and follow up",
        detail: "Go live, answer questions fast, then review results.",
      },
    ],
  },
  {
    type: "table",
    title: "Who owns each step",
    columns: ["Step", "Owner", "Due", "Budget"],
    rows: [
      ["Set offer, pricing and date", "You", "Week 1", "$0"],
      ["Build landing page", "Web or design", "Week 2", "$1,200"],
      ["Write launch emails and posts", "Marketing lead", "Week 2", "$300"],
      ["Test checkout and sign-up", "Operations", "Week 2", "$0"],
    ],
  },
  {
    type: "options",
    items: [
      { title: "Fix enrichment", description: "Recover the 31 lost leads" },
      { title: "Add a day 5 follow-up", tone: "positive" },
    ],
  },
];

const PART = {
  type: "tool-render_ui",
  toolCallId: "styleguide-render-ui",
  state: "output-available",
  input: {},
  output: { type: "ui_blocks", message: "Shown to the user.", blocks: BLOCKS },
} as unknown as ToolUIPart;

export function RenderUISamples() {
  return <RenderUI part={PART} canAct />;
}
