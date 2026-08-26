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
import { asObject, str } from "./resultHelpers";

interface Props {
  output: Record<string, unknown>;
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

/** An expert inside the chain — either a hire/raise preview awaiting the
 *  user's OK (given in chat, nothing created yet) or the teammate
 *  ``confirm_expert_change`` actually created. Once they exist, the card
 *  offers the two things the user does next: adjust them or talk to them. */
export function ExpertChangeCard({ output }: Props) {
  const applied = output.applied === true;
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
  const failed = applied ? failedWorkflows(output) : [];

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
        {applied ? (
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
      {applied && id && (
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
