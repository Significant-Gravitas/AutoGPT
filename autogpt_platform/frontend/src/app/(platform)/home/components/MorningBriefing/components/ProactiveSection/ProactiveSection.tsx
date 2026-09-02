"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  AiIdeaIcon,
  ArrowUpRight01Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useProactiveSection } from "./useProactiveSection";

export function ProactiveSection() {
  const { proposals, outcomes } = useProactiveSection();

  if (proposals.length === 0 && outcomes.length === 0) return null;

  return (
    <section
      aria-label="Proactive"
      className="flex flex-col gap-1 border-t border-zinc-100 pt-3"
    >
      <div className="flex items-center gap-1.5 pb-1">
        <Icon icon={AiIdeaIcon} size={15} className="text-fuchsia-600" />
        <Text variant="small" className="font-medium text-zinc-900">
          Proactive
        </Text>
      </div>
      {proposals.map((task) => (
        <ProactiveRow key={task.id} task={task} kind="proposal" />
      ))}
      {outcomes.map((task) => (
        <ProactiveRow key={task.id} task={task} kind="outcome" />
      ))}
    </section>
  );
}

interface ProactiveRowProps {
  task: DelegatedTask;
  kind: "proposal" | "outcome";
}

function ProactiveRow({ task, kind }: ProactiveRowProps) {
  return (
    <Link
      href={`/team/tasks/${task.id}`}
      className="group -mx-2 flex items-center gap-2.5 rounded-xl px-2 py-1.5 hover:bg-zinc-50"
    >
      {kind === "outcome" ? (
        <Icon
          icon={CheckmarkCircle02Icon}
          size={15}
          className="shrink-0 text-emerald-500"
        />
      ) : (
        <span className="mx-[3px] size-2 shrink-0 rounded-full bg-fuchsia-400" />
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm text-zinc-800">
          {task.title}
        </span>
        <span className="block truncate text-xs text-zinc-500">
          {kind === "outcome"
            ? (task.outcome_summary ?? "Done — open to see the outcome.")
            : "Suggested by your team — open to review."}
        </span>
      </span>
      <Icon
        icon={ArrowUpRight01Icon}
        size={14}
        className="shrink-0 text-zinc-300 transition-colors group-hover:text-zinc-600"
      />
    </Link>
  );
}
