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
  heldReview,
  mail,
  referenceCard,
  shell,
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

export const IrreversibleWithInputs: Story = { args: queueOf([mail()]) };

export const SupervisorCouldNotVouch: Story = { args: queueOf([shell()]) };

export const IdentifiedOnlyById: Story = {
  args: queueOf([deleteFolder("f1", "f-111"), deleteFolder("f2", "f-222")]),
};

// Ids the server resolved to names and pages when the call was held.
export const NamedFolder: Story = {
  args: queueOf([referenceCard("Delete folder")]),
};

export const NamedAgentList: Story = {
  args: queueOf([referenceCard("Move agents")]),
};

export const NamedSchedule: Story = {
  args: queueOf([referenceCard("Pause schedule")]),
};

export const NamedTemplate: Story = {
  args: queueOf([referenceCard("Hire expert")]),
};

export const NamedChat: Story = {
  args: queueOf([referenceCard("Message chat")]),
};

export const NamedCredential: Story = {
  args: queueOf([referenceCard("Grant credential")]),
};

export const NamedFile: Story = {
  args: queueOf([referenceCard("Delete file")]),
};

export const UnresolvedId: Story = {
  args: queueOf([referenceCard("Unresolved id")]),
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
