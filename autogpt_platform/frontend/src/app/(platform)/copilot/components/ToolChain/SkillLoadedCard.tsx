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
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon icon={BookBookmarkIcon} size={15} className="text-zinc-600" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-800">
          Skill loaded: {name} v{version} · {origin}
        </p>
        <p className="truncate text-xs text-zinc-500">
          Loaded for this task; loading is not a success record.
        </p>
      </div>
      <Link
        href={href}
        className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-600 hover:bg-zinc-200"
      >
        View change
      </Link>
    </div>
  );
}
