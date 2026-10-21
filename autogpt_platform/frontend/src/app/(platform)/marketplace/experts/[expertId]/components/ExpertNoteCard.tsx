import { Icon } from "@/components/atoms/Icon/Icon";
import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";

interface Props {
  icon: IconSvgElement;
  children: ReactNode;
}

/** A soft card for the page's facts about a hire — plan, rules — so they
 *  read as terms rather than as more of the essay above them. */
export function ExpertNoteCard({ icon, children }: Props) {
  return (
    <div className="flex gap-4 rounded-2xl bg-zinc-50 p-5 ring-1 ring-inset ring-zinc-200/60">
      <span
        aria-hidden="true"
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-inset ring-zinc-200/70"
      >
        <Icon icon={icon} size={20} className="text-zinc-600" />
      </span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
