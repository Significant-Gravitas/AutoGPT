import { TaskAmendment } from "@/app/api/__generated__/models/taskAmendment";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { UserIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import type { TaskActor } from "../../../../../task-helpers";

interface Props {
  at: TaskAmendment["at"];
  /** Event-kind glyph on the rail — the colored badge that lets the feed
   *  be scanned by shape (handoff, question, outcome) before reading. */
  icon: IconSvgElement;
  iconClassName: string;
  actor: TaskActor;
  href?: string;
  isLast: boolean;
  header: React.ReactNode;
  children?: React.ReactNode;
}

/** One stop on the feed: a kind badge on the left rail with a hairline
 *  dropping to the next stop; the actor's avatar and a one-line header
 *  ("<who> <did what> • <when>") with any quoted content beneath. */
export function ActivityEntry({
  at,
  icon,
  iconClassName,
  actor,
  href,
  isLast,
  header,
  children,
}: Props) {
  const headerRow = (
    <span className="flex min-h-6 min-w-0 flex-wrap items-center gap-x-1.5 text-[13px]">
      <EntryAvatar actor={actor} />
      {header}
      <span className="text-zinc-300">•</span>
      <time className="text-[11px] text-zinc-400">
        {new Date(at).toLocaleString(undefined, {
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit",
        })}
      </time>
    </span>
  );

  return (
    <li className="flex items-stretch gap-2.5">
      <div className="flex w-6 flex-col items-center pt-0.5">
        <span
          className={cn(
            "flex size-6 shrink-0 items-center justify-center rounded-full",
            iconClassName,
          )}
        >
          <Icon icon={icon} size={13} />
        </span>
        {!isLast && <div className="mt-1.5 w-px flex-1 bg-zinc-200" />}
      </div>
      <div className={`min-w-0 flex-1 ${isLast ? "pb-0" : "pb-5"}`}>
        {href ? (
          <Link
            href={href}
            className="-mx-1.5 inline-flex max-w-full rounded-lg px-1.5 transition-colors hover:bg-zinc-100"
          >
            {headerRow}
          </Link>
        ) : (
          headerRow
        )}
        {children}
      </div>
    </li>
  );
}

/** Autopilot's own work is stamped with the AutoGPT mark, the same way the
 *  copilot thread header and the task board mark an ownerless task. */
function EntryAvatar({ actor }: { actor: TaskActor }) {
  if (actor.kind === "expert") {
    return (
      <ExpertAvatar name={actor.name} avatarUrl={actor.avatarUrl} size={20} />
    );
  }

  return (
    <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-zinc-100 ring-1 ring-inset ring-zinc-200">
      {actor.kind === "autopilot" ? (
        <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-3" />
      ) : (
        <Icon icon={UserIcon} size={11} className="text-zinc-500" />
      )}
    </span>
  );
}
