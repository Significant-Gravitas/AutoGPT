import Link from "next/link";
import type { ReactNode } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import type { Reference } from "../helpers";

interface Props {
  reference: Reference & { href: string };
  children: ReactNode;
}

// With a summary the hover carries it and the id; without, the id is the title.
export function ReferenceLink({ reference, children }: Props) {
  const link = (
    <Link
      href={reference.href}
      translate="no"
      title={reference.summary ? undefined : reference.id}
      className="underline decoration-zinc-300 underline-offset-2 hover:decoration-zinc-500 focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
    >
      {children}
    </Link>
  );
  if (!reference.summary) return link;
  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>{link}</TooltipTrigger>
      <TooltipPortal>
        <TooltipContent side="bottom" align="start">
          <p className="text-zinc-900">{reference.summary}</p>
          <p className="mt-0.5 font-mono text-[0.6875rem] text-zinc-400">
            {reference.id}
          </p>
        </TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
