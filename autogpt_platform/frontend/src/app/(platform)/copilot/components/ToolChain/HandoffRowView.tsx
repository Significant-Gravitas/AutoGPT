"use client";

import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useDelegationAnswerStore } from "../../delegationAnswerStore";
import { delegationName } from "../../delegationViews";
import { useExpertMap } from "../../useExpertMap";
import { ChainRowBody } from "./ChainRowBody";
import { describeHandoffRow, handoffNodeOf } from "./delegationRow";
import { HandoffLoadingNode } from "./HandoffLoadingNode";
import { HandoffQuestionNode } from "./HandoffQuestionNode";
import type { ChainRow } from "./helpers";
import { asObject, str } from "./resultHelpers";

interface Props {
  row: ChainRow;
  isLast: boolean;
  readOnly: boolean;
}

function AnsweredRow({ text, isLast }: { text: string; isLast: boolean }) {
  return (
    <div className="flex items-stretch gap-2.5" data-testid="handoff-answered">
      <div className="flex w-7 flex-col items-center">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
          <Icon icon={Tick02Icon} size={14} className="text-zinc-600" />
        </div>
        {!isLast && <div className="w-px flex-1 bg-zinc-200" />}
      </div>
      <p
        className={
          "flex min-h-7 min-w-0 items-center text-sm text-zinc-600" +
          (isLast ? "" : " pb-3")
        }
      >
        <span className="truncate">You answered: {text}</span>
      </p>
    </div>
  );
}

/** A hand-off's row plus whatever it hangs on the wire: a skeleton while
 *  the call is on its way, the teammate's question while they wait on the
 *  user, the user's answer once sent. */
export function HandoffRowView({ row, isLast, readOnly }: Props) {
  const { expertsById } = useExpertMap();
  const delegation = row.delegation?.data;
  const subSessionId = delegation?.subSessionId ?? null;
  const answer = useDelegationAnswerStore((s) =>
    subSessionId ? (s.answers[subSessionId] ?? null) : null,
  );
  const expertId = str(asObject(row.input) ?? {}, "expert_id");
  const name = delegation
    ? delegationName(delegation, expertsById)
    : ((expertId && expertsById.get(expertId)?.name) ?? "a teammate");
  const node = handoffNodeOf(row, readOnly, !!answer);
  const shown = describeHandoffRow(row, name);

  return (
    <div className="flex flex-col">
      <ChainRowBody
        row={shown}
        isLast={node === "answered" ? false : node ? true : isLast}
        readOnly={readOnly}
      />
      {node === "loading" && (
        <HandoffLoadingNode isLast={isLast} expertName={name} />
      )}
      {node === "question" && delegation && (
        <HandoffQuestionNode delegation={delegation} isLast={isLast} />
      )}
      {node === "answered" && answer && (
        <AnsweredRow text={answer.text} isLast={isLast} />
      )}
    </div>
  );
}
