"use client";

import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  HelpCircleIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { CARD } from "./ResultCards";
import { asObject, str } from "./resultHelpers";

interface Props {
  output: Record<string, unknown>;
}

interface VerdictStyle {
  label: string;
  icon: IconSvgElement;
  chip: string;
}

const VERDICTS: Record<string, VerdictStyle> = {
  pass: {
    label: "No objection",
    icon: CheckmarkCircle02Icon,
    chip: "bg-green-50 text-green-700",
  },
  block: {
    label: "Blocked",
    icon: Alert02Icon,
    chip: "bg-red-50 text-red-600",
  },
  insufficient: {
    label: "Not checked",
    icon: HelpCircleIcon,
    chip: "bg-amber-50 text-amber-700",
  },
};

/** One teammate's ruling on another's draft. The quoted lines are the point of
 *  the card: they are what the user reads to decide whether the objection was
 *  right, and whether an override in the reply below it was honest.
 *
 *  `reason` and `quotes` are model output conditioned on a user-editable Soul,
 *  so they render as plain text — never markdown, never HTML. */
export function ConsultVerdictCard({ output }: Props) {
  const verdict = str(output, "verdict") ?? "";
  const style = VERDICTS[verdict];
  const reviewer = asObject(output.reviewer);
  if (!style || !reviewer) return null;

  const quotes = Array.isArray(output.quotes)
    ? output.quotes.filter(
        (q): q is string => typeof q === "string" && !!q.trim(),
      )
    : [];
  const reason = str(output, "reason");

  return (
    <div className={cn(CARD, "w-full rounded-3xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={str(reviewer, "name") ?? null}
          avatarUrl={str(reviewer, "avatar_url") ?? null}
          size={28}
        />
        <div className="min-w-0 flex-1">
          <Text variant="body-medium" className="truncate text-zinc-800">
            {str(reviewer, "name") ?? "A teammate"}
          </Text>
          <Text variant="small" className="truncate text-zinc-500">
            {str(reviewer, "role") ?? "checked this"}
          </Text>
        </div>
        <span
          className={cn(
            "flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5",
            style.chip,
          )}
        >
          <Icon icon={style.icon} size={14} />
          <Text variant="small-medium">{style.label}</Text>
        </span>
      </div>
      {reason ? (
        <Text variant="small" className="mt-2 px-0.5 text-zinc-600">
          {reason}
        </Text>
      ) : null}
      {quotes.length > 0 ? (
        <ul className="mt-2 space-y-1">
          {quotes.map((quote) => (
            <li
              key={quote}
              className="border-l-2 border-zinc-200 pl-2 text-zinc-500"
            >
              <Text variant="small" className="text-zinc-500">
                {quote}
              </Text>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
