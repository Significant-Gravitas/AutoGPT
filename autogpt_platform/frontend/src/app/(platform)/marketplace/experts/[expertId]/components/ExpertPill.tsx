import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { ReactNode } from "react";

interface Props {
  icon: ReactNode;
  label: string;
  href?: string;
}

const PILL_CLASS =
  "flex min-w-0 items-center gap-2 rounded-lg bg-white px-2.5 py-1.5 text-sm text-zinc-700 ring-1 ring-inset ring-zinc-200/80";

/** The flat item the expert page lists services and skills with. */
export function ExpertPill({ icon, label, href }: Props) {
  // The label names the item, so a screen reader must not hear the icon too.
  const content = (
    <>
      <span aria-hidden="true" className="flex shrink-0">
        {icon}
      </span>
      <span className="truncate">{label}</span>
    </>
  );

  if (!href) return <li className={PILL_CLASS}>{content}</li>;

  return (
    <li className="min-w-0">
      <Link
        href={href}
        className={cn(
          PILL_CLASS,
          "outline-none transition-colors hover:text-zinc-900 hover:ring-zinc-300 focus-visible:ring-2 focus-visible:ring-violet-600",
        )}
      >
        {content}
        <Icon
          icon={ArrowUpRight01Icon}
          size={14}
          aria-hidden
          className="shrink-0 text-zinc-400"
        />
      </Link>
    </li>
  );
}
