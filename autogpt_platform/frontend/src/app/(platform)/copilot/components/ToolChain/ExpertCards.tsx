"use client";

import {
  ArrowDown01Icon,
  BubbleChatIcon,
  CheckmarkCircle02Icon,
  PencilIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useId, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { CARD } from "./ResultCards";
import { asObject, str } from "./resultHelpers";
import { useCardResize } from "./useCardResize";

export const EXPERT_CHANGE_TOOLS = new Set([
  "hire_expert",
  "raise_expert",
  "update_expert",
  "confirm_expert_change",
]);

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
  const [expanded, setExpanded] = useState(false);
  const detailsID = useId();
  const { contentRef, height } = useCardResize(expanded);
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
  const clampable = !!about || !!boundaries;
  const clamp = expanded ? undefined : "line-clamp-2";

  return (
    <div className={cn(CARD, "w-full rounded-3xl p-2.5")}>
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
        {clampable && (
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
            aria-controls={detailsID}
            aria-label={expanded ? "Show less" : "Show more"}
            className="group/expand -mr-0.5 flex size-6 shrink-0 items-center justify-center rounded-full transition-colors hover:bg-zinc-100"
          >
            <Icon
              icon={ArrowDown01Icon}
              size={12}
              className={cn(
                "text-zinc-300 transition-transform duration-300 ease-out-quint group-hover/expand:text-zinc-500",
                expanded && "rotate-180",
              )}
            />
          </button>
        )}
      </div>
      <div
        id={detailsID}
        className="overflow-hidden transition-[height] duration-300 ease-out-quint motion-reduce:transition-none"
        style={{ height }}
      >
        <div ref={contentRef} className="flex flex-col gap-1 pt-1.5">
          {about && (
            <p className={cn("pl-9 text-sm text-zinc-500", clamp)}>{about}</p>
          )}
          {boundaries && (
            <p className={cn("pl-9 text-sm text-zinc-400", clamp)}>
              Stops at: {boundaries}
            </p>
          )}
          {budget !== null && (
            <p className="pl-9 text-sm text-zinc-600">
              Weekly budget: {budget} credits
            </p>
          )}
          {failed.length > 0 && (
            <p className="pl-9 text-[11px] text-amber-700">
              Couldn&apos;t set up: {failed.join(", ")}. Everything else is
              ready — add {failed.length > 1 ? "them" : "it"} from the
              expert&apos;s page.
            </p>
          )}
          {applied && id && (
            <div className="mt-1.5 flex items-center gap-1.5 pl-9">
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
      </div>
    </div>
  );
}

export function ExpertChangeCardSkeleton() {
  return (
    <div className={cn(CARD, "w-full rounded-3xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <Skeleton className="size-7 shrink-0 rounded-full" />
        <Skeleton className="h-3.5 w-40" />
        <Skeleton className="ml-auto h-4 w-20 rounded-md" />
      </div>
      <div className="flex flex-col gap-1.5 pl-9 pt-2.5">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-4/5" />
      </div>
    </div>
  );
}
