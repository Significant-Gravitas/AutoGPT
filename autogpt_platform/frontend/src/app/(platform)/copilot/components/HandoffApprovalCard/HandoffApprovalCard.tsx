"use client";

import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { CardStatus } from "../ApprovalQueue/components/ApprovalCard/ApprovalCard";
import type {
  ApprovalItem,
  ChatRule,
  RuleScope,
} from "../ApprovalQueue/helpers";
import { useExpertMap } from "../../useExpertMap";
import { HandoffActions } from "./components/HandoffActions";
import { HandoffBrief } from "./components/HandoffBrief";
import { HandoffMeta } from "./components/HandoffMeta";
import { capLine, type HandoffEdits, readHandoff } from "./helpers";
import { useHandoffApprovalCard } from "./useHandoffApprovalCard";

interface Props {
  item: ApprovalItem;
  payload: unknown;
  editable: boolean;
  status: CardStatus;
  failed: boolean;
  onApprove: (edits: HandoffEdits, rule?: ChatRule, scope?: RuleScope) => void;
  onReject: () => void;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium uppercase leading-4 tracking-[0.06em] text-zinc-500">
        {title}
      </span>
      {children}
    </div>
  );
}

/** Ask First's hand-off card: who gets the work, why, the brief they will
 *  read, and what comes back — answered without leaving the chat. */
export function HandoffApprovalCard({
  item,
  payload,
  editable,
  status,
  failed,
  onApprove,
  onReject,
}: Props) {
  const { expertsById, activeExperts } = useExpertMap();
  const facts = readHandoff(item, payload, expertsById);
  const card = useHandoffApprovalCard(facts.brief);
  const picked = card.pickedExpertId
    ? expertsById.get(card.pickedExpertId)
    : undefined;
  const expert = picked
    ? { ...picked, color: picked.color ?? null }
    : facts.expert;
  const nameWithRole = expert.role
    ? `${expert.name} · ${expert.role}`
    : expert.name;

  return (
    <article aria-busy={status !== "idle"} className="flex flex-col">
      <header className="flex items-center gap-3 border-b border-zinc-100 bg-zinc-50 px-4 py-3">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatarUrl}
          color={expert.color}
          size={28}
        />
        <p className="min-w-0 flex-1 text-sm leading-[22px] text-zinc-900">
          Hand off “{facts.title}” to {nameWithRole}
        </p>
        <span className="shrink-0 text-xs leading-[18px] text-zinc-500">
          Ask First
        </span>
      </header>
      <div className="flex flex-col gap-3 px-4 py-3.5">
        {failed && (
          <p role="alert" className="text-sm text-red-600">
            Couldn&apos;t send your answer. Nothing ran. Try again.
          </p>
        )}
        {facts.why && (
          <Section title={`Why ${expert.name}`}>
            <p className="text-sm leading-[22px] text-zinc-900">{facts.why}</p>
          </Section>
        )}
        {(facts.brief || card.isEditing) && (
          <Section title="Brief">
            <HandoffBrief
              brief={facts.brief ?? ""}
              isEditing={card.isEditing}
              draft={card.draft}
              onDraftChange={card.setDraft}
              isExpanded={card.isExpanded}
              onToggleExpanded={card.toggleExpanded}
            />
          </Section>
        )}
        <HandoffMeta
          facts={[
            { label: "Expected back", value: facts.expectedBack },
            { label: "By", value: facts.by },
            { label: "Cap", value: capLine(item) },
            { label: "Owner", value: "You" },
          ]}
        />
      </div>
      <HandoffActions
        status={status}
        expertName={expert.name}
        canEdit={editable}
        isEditing={card.isEditing}
        canAlwaysAllow={
          !card.pickedExpertId &&
          !item.spend &&
          item.chatRulesAllowed.includes("allow")
        }
        otherExperts={activeExperts.filter((e) => e.id !== expert.id)}
        onApprove={() => onApprove(card.edits)}
        onReject={onReject}
        onToggleEdit={card.toggleEditing}
        onPickExpert={card.pickExpert}
        onAlwaysAllow={() => onApprove({}, "allow", "expert")}
      />
    </article>
  );
}
