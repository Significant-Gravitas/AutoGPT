"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight01Icon, UserGroupIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";

interface Props {
  task: DelegatedTask;
}

export function TaskTopBar({ task }: Props) {
  return (
    <div className="sticky top-0 z-20 flex items-center gap-4 border-b-[0.5px] border-zinc-200 bg-white/85 px-6 py-3 backdrop-blur md:px-8">
      <nav
        aria-label="Breadcrumb"
        className="flex min-w-0 items-center gap-1.5 text-sm"
      >
        <Link
          href="/team"
          className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-800"
          data-testid="task-back-to-team"
        >
          <Icon icon={UserGroupIcon} size={14} className="text-zinc-400" />
          Team
        </Link>
        <Icon icon={ArrowRight01Icon} size={14} className="text-zinc-300" />
        <span className="truncate text-zinc-900">{task.title}</span>
      </nav>
    </div>
  );
}
