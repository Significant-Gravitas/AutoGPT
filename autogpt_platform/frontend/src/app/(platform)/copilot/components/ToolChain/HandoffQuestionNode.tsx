"use client";

import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { type ChatDelegation, formatElapsed } from "../../delegations";
import { delegationTitle, threadHref } from "../../delegationViews";
import { useDelegationAnswer } from "../../useDelegationAnswer";
import { useDelegationLive } from "../../useDelegationLive";
import { DelegationQuestionBox } from "../DelegationQuestion/DelegationQuestionBox";
import { WireNode } from "./WireNode";

interface Props {
  delegation: ChatDelegation;
  isLast: boolean;
}

/** A teammate stopped on a question: it hangs on Otto's wire under the
 *  hand-off, so the user answers without leaving this chat. */
export function HandoffQuestionNode({ delegation, isLast }: Props) {
  const live = useDelegationLive(delegation);
  const { sendAnswer, isSending } = useDelegationAnswer();
  const { expert } = live;
  const href = threadHref(delegation);
  const waited = live.askedAt
    ? formatElapsed((Date.now() - live.askedAt) / 1000)
    : null;
  const title = delegationTitle(delegation, expert.name);
  const heading = [`${expert.name} asks`, `paused on “${title}”`, waited]
    .filter(Boolean)
    .join(" · ");
  const question = live.question ?? delegation.question;
  if (!question || !delegation.subSessionId) return null;
  const subSessionId = delegation.subSessionId;

  return (
    <WireNode isLast={isLast} testId="handoff-question-node" tone="waiting">
      <div className="flex items-center gap-3 border-b border-zinc-100 bg-amber-50 px-4 py-3">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatarUrl}
          color={expert.color}
          size={28}
        />
        <p className="min-w-0 flex-1 text-sm leading-[22px] text-zinc-900">
          {heading}
        </p>
        <span className="shrink-0 rounded-md bg-amber-50 px-2 py-0.5 text-xs font-medium leading-5 text-amber-800 ring-1 ring-inset ring-amber-500/20">
          Needs you
        </span>
      </div>
      <div className="px-4 py-3.5">
        <DelegationQuestionBox
          question={question}
          options={live.questionOptions}
          expertName={expert.name}
          variant="chain"
          isSending={isSending}
          onSend={(answer) => void sendAnswer(subSessionId, answer, question)}
        />
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-zinc-100 px-4 py-2.5">
        {href && (
          <Link
            href={href}
            className="inline-flex h-9 items-center gap-1.5 rounded-full px-3 text-sm font-medium text-zinc-900 hover:bg-zinc-50"
          >
            Open {expert.name}&apos;s thread
            <Icon icon={ArrowRight01Icon} size={16} />
          </Link>
        )}
        <span className="flex-1" />
        <span className="text-xs text-zinc-500">
          Your answer continues {expert.name}&apos;s thread
        </span>
      </div>
    </WireNode>
  );
}
