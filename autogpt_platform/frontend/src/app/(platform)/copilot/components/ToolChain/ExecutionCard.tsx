"use client";

import { ArrowSquareOutIcon, PlayIcon, RobotIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { CARD, HALF, StatusPill } from "./ResultCards";

export function ExecutionCard({
  name,
  status,
  href,
  variant = "run",
}: {
  name: string;
  status?: string;
  href?: string;
  variant?: "run" | "agent";
}) {
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        {variant === "agent" ? (
          <RobotIcon size={15} className="text-zinc-600" />
        ) : (
          <PlayIcon size={13} weight="fill" className="text-zinc-600" />
        )}
      </div>
      <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-800">
        {name}
      </p>
      {status && <StatusPill status={status} />}
      {href && (
        <Link
          href={href}
          aria-label={variant === "agent" ? "Open agent" : "Open execution"}
          className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <ArrowSquareOutIcon size={14} />
        </Link>
      )}
    </div>
  );
}
