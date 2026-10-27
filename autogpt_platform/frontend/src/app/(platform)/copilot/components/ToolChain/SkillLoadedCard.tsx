"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { skillDetailHref } from "@/services/skill-learning/helpers";
import { BookBookmarkIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { CARD, HALF } from "./ResultCards";
import { str } from "./resultHelpers";

interface Props {
  output: Record<string, unknown>;
}

export function SkillLoadedCard({ output }: Props) {
  const name = str(output, "name");
  const version = output.version;
  if (!name || typeof version !== "number") return null;
  const origin = str(output, "origin_label") ?? "Saved";
  const href = skillDetailHref({
    expertId: str(output, "expert_id"),
    skillName: name,
    versionId: str(output, "version_id"),
  });
  return (
    <div
      className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}
      data-testid="skill-loaded-card"
    >
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
        <Icon
          icon={BookBookmarkIcon}
          size={15}
          className="text-foreground-secondary"
        />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
          Skill loaded: {name} v{version} · {origin}
        </p>
        <p className="truncate text-xs text-foreground-tertiary">
          Loaded for this task; loading is not a success record.
        </p>
      </div>
      <Link
        href={href}
        className="shrink-0 rounded-full bg-surface-hover px-2 py-0.5 text-caption-sm leading-inherit text-foreground-secondary hover:bg-surface-active"
      >
        View change
      </Link>
    </div>
  );
}
