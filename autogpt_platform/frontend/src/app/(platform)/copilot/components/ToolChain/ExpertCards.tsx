"use client";

import {
  BubbleChatIcon,
  CheckmarkCircle02Icon,
  PencilIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { CARD, HALF } from "./ResultCards";
import { asItems, asObject, str } from "./resultHelpers";

interface Props {
  output: Record<string, unknown>;
  applied?: boolean;
}

const APPLIED_LABELS: Record<string, string> = {
  hire: "Hired",
  raise: "Raised",
  update: "Updated",
};

function failedWorkflows(output: Record<string, unknown>): string[] {
  const value = output.failed_workflows;
  if (!Array.isArray(value)) return [];
  return value.filter(
    (entry): entry is string => typeof entry === "string" && !!entry.trim(),
  );
}

/** One or several experts inside the chain. A batched confirm carries a
 *  ``results`` list — one entry per approved preview, each either the
 *  teammate it created or why it didn't — so a single bad id in the batch
 *  never hides the ones that landed. */
export function ExpertChangeCard({ output }: Props) {
  const results = asItems(output.results);
  if (!results) return <ExpertRow output={output} />;

  return (
    <div className="flex flex-col gap-1.5">
      {results.map((result, index) => (
        <ResultRow key={resultKey(result, index)} result={result} />
      ))}
    </div>
  );
}

function resultKey(result: Record<string, unknown>, index: number): string {
  return str(result, "confirmation_id") ?? `result-${index}`;
}

interface ResultProps {
  result: Record<string, unknown>;
}

/** ``outcome`` is three-valued: an approval that landed in an earlier
 *  confirm is done, not a failure, and drawing it as "Not added" tells the
 *  user a teammate they have was never created. */
function ResultRow({ result }: ResultProps) {
  const outcome = str(result, "outcome");
  if (outcome === "applied") return <ExpertRow output={result} applied />;
  return (
    <ExpertNoticeRow result={result} done={outcome === "already_applied"} />
  );
}

interface NoticeProps {
  result: Record<string, unknown>;
  done: boolean;
}

function ExpertNoticeRow({ result, done }: NoticeProps) {
  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      <p className="flex items-center gap-1.5 text-[13px] font-medium text-zinc-800">
        {done && (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={13}
            className="text-emerald-600"
          />
        )}
        {done ? "Already done" : "Not added"}
      </p>
      <p className="mt-1 text-xs text-zinc-500">
        {str(result, "error") ?? "This approval could no longer be applied."}
      </p>
    </div>
  );
}

/** An expert inside the chain — either a hire/raise preview awaiting the
 *  user's OK (given in chat, nothing created yet) or the teammate
 *  ``confirm_expert_change`` actually created. Once they exist, the card
 *  offers the two things the user does next: adjust them or talk to them. */
function ExpertRow({ output, applied }: Props) {
  const isApplied = applied ?? output.applied === true;
  const expert = asObject(output.expert) ?? asObject(output.preview);
  if (!expert) return null;

  const id = str(expert, "id");
  const name = str(expert, "name") ?? "New expert";
  const role = str(expert, "role");
  const about = str(expert, "about");
  const boundaries = str(expert, "boundaries");
  const budget =
    typeof expert.weekly_budget === "number" ? expert.weekly_budget : null;
  const appliedLabel = APPLIED_LABELS[str(output, "kind") ?? ""] ?? "Done";
  const failed = isApplied ? failedWorkflows(output) : [];

  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={name}
          avatarUrl={str(expert, "avatar_url")}
          size={28}
        />
        <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-800">
          {name}
          {role && (
            <span className="ml-1.5 font-normal text-zinc-400">{role}</span>
          )}
        </p>
        {isApplied ? (
          <span className="inline-flex shrink-0 items-center gap-1 rounded-md bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
            <Icon icon={CheckmarkCircle02Icon} size={11} />
            {appliedLabel}
          </span>
        ) : (
          <span className="shrink-0 rounded-md bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700">
            Needs your OK
          </span>
        )}
      </div>
      {about && (
        <p className="mt-1.5 line-clamp-2 pl-9 text-xs text-zinc-500">
          {about}
        </p>
      )}
      {boundaries && (
        <p className="mt-1 line-clamp-2 pl-9 text-xs text-zinc-400">
          Stops at: {boundaries}
        </p>
      )}
      {budget !== null && (
        <p className="mt-1 pl-9 text-[11px] text-zinc-400">
          Weekly budget: {budget} credits
        </p>
      )}
      {failed.length > 0 && (
        <p className="mt-1 pl-9 text-[11px] text-amber-700">
          Couldn&apos;t set up: {failed.join(", ")}. Everything else is ready —
          add {failed.length > 1 ? "them" : "it"} from the expert&apos;s page.
        </p>
      )}
      {isApplied && id && (
        <div className="mt-2.5 flex items-center gap-1.5 pl-9">
          <Link
            href={`/team/${id}`}
            className="inline-flex h-7 items-center gap-1.5 rounded-full border border-zinc-200 px-3 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-50 hover:text-zinc-900"
          >
            <Icon icon={PencilIcon} size={13} />
            Edit
          </Link>
          <Link
            href={`/copilot?expertId=${id}`}
            className="inline-flex h-7 items-center gap-1.5 rounded-full bg-zinc-900 px-3 text-xs font-medium text-white transition-colors hover:bg-zinc-700"
          >
            <Icon icon={BubbleChatIcon} size={13} />
            Chat
          </Link>
        </div>
      )}
    </div>
  );
}
