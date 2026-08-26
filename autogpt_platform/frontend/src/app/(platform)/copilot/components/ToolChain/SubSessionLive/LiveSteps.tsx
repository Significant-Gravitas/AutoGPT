import { cn } from "@/lib/utils";
import Link from "next/link";
import type { ChainRow } from "../helpers";
import { RowIcon } from "../RowIcon";
import { SwapText } from "../SwapText";

export function LiveSteps({ rows }: { rows: ChainRow[] }) {
  return rows.map((row, i) => {
    const isLastRow = i === rows.length - 1;
    return (
      <div key={row.key} className="flex items-stretch gap-2.5">
        <div className="flex w-6 flex-col items-center">
          <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-zinc-100">
            <RowIcon row={row} />
          </div>
          {!isLastRow && <div className="w-px flex-1 bg-zinc-200" />}
        </div>
        <div className={cn("min-w-0 flex-1 pt-[2px]", !isLastRow && "pb-2.5")}>
          <SwapText
            text={row.text}
            shimmer={row.state === "running"}
            className="max-w-full text-sm leading-5 text-zinc-600"
          />
        </div>
      </div>
    );
  });
}

/** Says the polling stopped and keeps the deep link, so the user can follow
 *  the run at its source instead of watching a spinner that never resolves. */
export function LiveNotice({
  text,
  subSessionId,
}: {
  text: string;
  subSessionId: string;
}) {
  return (
    <p className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-zinc-400">
      {text}
      <Link
        href={`/copilot?sessionId=${subSessionId}`}
        className="underline underline-offset-2 hover:text-zinc-600"
      >
        Open sub-session
      </Link>
    </p>
  );
}
