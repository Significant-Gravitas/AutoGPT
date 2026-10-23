import Link from "next/link";
import type { ReactNode } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import type { Reference } from "../helpers";
import { ReferenceCard } from "./ReferenceCard";

interface Props {
  reference: Reference;
  children: ReactNode;
}

// With a card the hover carries it and the id; without, the id is the title.
// A thing with no page of its own is named without a link, card and all.
export function ReferenceLink({ reference, children }: Props) {
  // A reference stored before cards had kinds carries only its summary line.
  const hasCard = Boolean(reference.kind || reference.summary);
  const link = reference.href ? (
    <Link
      href={reference.href}
      translate="no"
      title={hasCard ? undefined : reference.id}
      className="underline decoration-zinc-300 underline-offset-2 hover:decoration-zinc-500 focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
    >
      {children}
    </Link>
  ) : (
    <span
      translate="no"
      // Focusable only when it opens a card, so the keyboard reaches it too.
      tabIndex={hasCard ? 0 : undefined}
      title={hasCard ? undefined : reference.id}
      className={
        hasCard
          ? "underline decoration-zinc-400 decoration-dotted underline-offset-2 focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
          : undefined
      }
    >
      {children}
    </span>
  );
  if (!hasCard) return link;
  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>{link}</TooltipTrigger>
      <TooltipPortal>
        <TooltipContent
          side="bottom"
          align="start"
          className="w-max max-w-[320px] px-3 py-2.5"
        >
          <ReferenceCard reference={reference} />
        </TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
