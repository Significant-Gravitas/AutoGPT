import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import type { HeldOutcome } from "../ChatMessagesContainer/heldCallRows";
import { ChainRowView } from "../ToolChain/ChainRowView";
import { applyHeldOutcome } from "../ToolChain/heldRow";
import { toChainRow } from "../ToolChain/helpers";
import { AttentionRow } from "../../../home/components/NeedsYou/components/AttentionRow";
import { ApprovalQueue } from "./ApprovalQueue";
import { toApprovalItem } from "./helpers";
import {
  deleteFolder,
  folder,
  heldRead,
  heldReview,
  mail,
  mcpTool,
  realCardSchemaHandler,
  realCards,
  shell,
  workflow,
} from "./__tests__/fixtures";

function answerAfter(ms: number, status = 200) {
  return http.post("*/api/review/action", async () => {
    await delay(ms);
    return status === 200
      ? HttpResponse.json({
          approved_count: 1,
          rejected_count: 0,
          failed_count: 0,
        })
      : HttpResponse.json({ detail: "unavailable" }, { status });
  });
}

function queueOf(reviews: PendingHumanReviewModel[]) {
  return { items: reviews.map(toApprovalItem), onAnswered: () => undefined };
}

const meta: Meta<typeof ApprovalQueue> = {
  title: "Copilot/ApprovalQueue",
  component: ApprovalQueue,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="mx-auto w-full max-w-[42rem]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  parameters: { msw: { handlers: [answerAfter(600_000)] } },
};

export default meta;
type Story = StoryObj<typeof ApprovalQueue>;

export const OneCard: Story = { args: queueOf([folder("a", "Q3 reports")]) };

export const TwoOnTheSameSubject: Story = {
  args: queueOf([folder("a", "Q3 reports", 6), folder("b", "Invoices", 5)]),
};

const GMAIL_SCHEMA = http.get("*/api/builder/blocks/batch", () =>
  HttpResponse.json([
    {
      id: "b-gmail",
      name: "GmailSendBlock",
      inputSchema: {
        type: "object",
        required: ["to", "subject"],
        properties: {
          to: { type: "array", title: "To" },
          subject: { type: "string", title: "Subject" },
          body: { type: "string", title: "Body" },
        },
      },
    },
  ]),
);

export const IrreversibleWithInputs: Story = {
  args: queueOf([mail()]),
  parameters: { msw: { handlers: [GMAIL_SCHEMA, answerAfter(600_000)] } },
};

// As the server sends it: no chat rule is offered for a block yet.
export const BlockCard: Story = {
  args: queueOf([mail("mail", [])]),
  parameters: { msw: { handlers: [GMAIL_SCHEMA, answerAfter(600_000)] } },
};

export const WorkflowRun: Story = { args: queueOf([workflow()]) };

// The Approve menu: the chat rules the server allows on this subject.
export const RuleMenu: Story = { args: queueOf([mcpTool()]) };

export const BlockChainRow: StoryObj = {
  render: () => {
    const part = {
      type: "tool-run_capability",
      state: "output-available",
      toolCallId: "call-gmail",
      input: { id: "b-gmail", input: { to: ["dana@acme.com"] } },
      output: {
        type: "approval_required",
        tool_name: "run_capability",
        review_id: "copilot-node-gate-run_capability:gmail",
        ask: "Run",
        object: "Gmail Send",
      },
    } as MessagePart;
    const row = applyHeldOutcome(toChainRow(part, 0)!, new Map());
    return <ChainRowView row={row} isLast />;
  },
};

export const SupervisorCouldNotVouch: Story = { args: queueOf([shell()]) };

export const IdentifiedOnlyById: Story = {
  args: queueOf([deleteFolder("f1", "f-111"), deleteFolder("f2", "f-222")]),
};

export const FiveWaiting: Story = {
  args: queueOf([
    heldReview({
      id: "d",
      tool: "delete_skill",
      args: { name: "Old drafts" },
      fields: [{ key: "name", label: "Name" }],
      headline: {
        ask: "Delete skill",
        object: "Old drafts",
        object_key: "name",
      },
      mode: "auto",
      reason:
        "This action reaches outside the platform, so it needs your approval.",
      minutesAgo: 9,
    }),
    mail("m", []),
    shell(),
    folder("f", "Receipts"),
    deleteFolder("x", "f-333"),
  ]),
};

export const SendFailed: Story = {
  args: queueOf([folder("a", "Q3 reports")]),
  parameters: { msw: { handlers: [answerAfter(300, 503)] } },
};

export const Narrow: Story = {
  args: queueOf([mail(), folder("b", "Invoices")]),
  parameters: { viewport: { defaultViewport: "mobile1" } },
};

const HELD_PART = (id: string, name: string): MessagePart =>
  ({
    type: "tool-create_folder",
    state: "output-available",
    toolCallId: `call-${id}`,
    input: { name },
    output: {
      type: "approval_required",
      tool_name: "create_folder",
      review_id: `copilot-node-gate-create_folder:${id}`,
      ask: "Create folder",
      object: name,
    },
  }) as MessagePart;

const OUTCOMES = new Map<string, HeldOutcome>([
  ["call-b", { outcome: "approved", output: { message: "Created" } }],
  ["call-c", { outcome: "rejected", output: "" }],
  ["call-d", { outcome: "expired", output: "" }],
]);

export const ChainRows: StoryObj = {
  render: () => {
    const parts = [
      HELD_PART("a", "Receipts"),
      HELD_PART("b", "Q3 reports"),
      HELD_PART("c", "Invoices"),
      HELD_PART("d", "Old drafts"),
      {
        ...HELD_PART("e", "Archive"),
        output: {
          type: "approval_required",
          tool_name: "create_folder",
          review_id: null,
          ask: "Create folder",
        },
      } as MessagePart,
    ];
    const rows = parts.map((part, i) =>
      applyHeldOutcome(toChainRow(part, i)!, OUTCOMES),
    );
    return (
      <div className="flex flex-col">
        {rows.map((row, i) => (
          <ChainRowView
            key={row.key}
            row={row}
            isLast={i === rows.length - 1}
          />
        ))}
      </div>
    );
  },
};

export const HeldRead: Story = {
  args: queueOf([heldRead("r", "docs.northwind.io/billing")]),
};

export const HeldReadBesideAnAction: Story = {
  args: queueOf([
    folder("a", "Q3 reports", 6),
    heldRead("r", "docs.northwind.io/billing"),
  ]),
};

// The judge failed (timeout, gateway error): no verdict, so no quote.
export const HeldReadUnchecked: Story = {
  args: queueOf([
    (() => {
      const review = heldRead("u", "status.acme.dev");
      const payload = review.payload as Record<string, unknown>;
      return { ...review, payload: { ...payload, judged: false, passage: "" } };
    })(),
  ]),
};

// Answered at once, so clicking leaves the Released / Kept out receipt.
export const HeldReadAnswered: Story = {
  args: queueOf([
    heldRead("a", "docs.northwind.io/billing"),
    heldRead("b", "pastebin.example/raw/x1"),
  ]),
  parameters: { msw: { handlers: [answerAfter(0)] } },
};

const READ_PART = (id: string, url: string) =>
  ({
    type: "tool-web_fetch",
    state: "output-available",
    toolCallId: `read-${id}`,
    input: { url },
    output: {
      type: "approval_required",
      tool_name: "web_fetch",
      review_id: `copilot-node-gate-read-web_fetch:${id}`,
      ask: "Read",
      object: url,
    },
  }) as MessagePart;

const READ_OUTCOMES = new Map<string, HeldOutcome>([
  ["read-b", { outcome: "approved", output: { message: "fetched" } }],
  ["read-c", { outcome: "rejected", output: "" }],
  ["read-d", { outcome: "unknown", output: "" }],
]);

export const HeldReadChainRows: StoryObj = {
  render: () => {
    const rows = [
      READ_PART("a", "docs.northwind.io/billing"),
      READ_PART("b", "status.acme.dev"),
      READ_PART("c", "pastebin.example/raw/x1"),
      READ_PART("d", "files.acme.dev/q3.csv"),
    ].map((part, i) => applyHeldOutcome(toChainRow(part, i)!, READ_OUTCOMES));
    return (
      <div className="flex flex-col">
        {rows.map((row, i) => (
          <ChainRowView
            key={row.key}
            row={row}
            isLast={i === rows.length - 1}
          />
        ))}
      </div>
    );
  },
};

function homeRow(
  title: string,
  description: string,
  action: string,
  headline?: { ask: string; object: string },
) {
  const review = folder("a", "Q3 reports");
  return (
    <div className="rounded-xl border border-zinc-200 bg-white">
      <AttentionRow
        item={{
          id: "approval-a",
          kind: "approval",
          priority: "normal",
          title,
          headline,
          description,
          why_it_matters: "Nothing runs until you approve it.",
          review,
          primary_action: { label: action, href: "/copilot?sessionId=s1" },
        }}
        isProcessing={false}
        onDecision={() => undefined}
      />
    </div>
  );
}

// What compose_attention_items returned for a gate row before and after this layer.
export const HomeRowBefore: StoryObj = {
  render: () =>
    homeRow(
      "Create folder — Ask First is on for this chat, so this action needs your approval.",
      "Otto is waiting for your approval.",
      "Review",
    ),
};

export const HomeRowAfter: StoryObj = {
  render: () =>
    homeRow(
      "Create folder “Q3 reports”",
      "Otto is waiting for your approval.",
      "Open chat",
      { ask: "Create folder", object: "Q3 reports" },
    ),
};

// Real registry blocks, their payloads built by the server's own builder and
// their real input schemas served as the API serves them.
function realStory(name: string): Story {
  const cards = realCards();
  const card = cards.find((c) => c.story === name)!;
  return {
    args: queueOf([card.review]),
    parameters: {
      msw: { handlers: [realCardSchemaHandler(cards), answerAfter(600_000)] },
    },
  };
}

export const RealGmailSend = realStory("Gmail Send");
export const RealGoogleSheetsUpdateRow = realStory("Google Sheets Update Row");
export const RealExecuteCodeStep = realStory("Execute Code Step");
export const RealSendWebRequest = realStory("Send Web Request");
export const RealPostToX = realStory("Post To X");
export const RealWorkflow = realStory("Workflow");
