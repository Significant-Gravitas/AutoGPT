import { cn } from "@/lib/utils";
import { DONE, IN_PROGRESS, SCHEDULED, TODO } from "../../helpers";

interface Props {
  columnId: string;
  className?: string;
}

export function StatusIcon({ columnId, className }: Props) {
  const classes = cn("size-3.5 shrink-0", className);

  if (columnId === SCHEDULED) {
    return (
      <svg viewBox="0 0 14 14" aria-hidden className={classes}>
        <circle
          cx="7"
          cy="7"
          r="6"
          fill="none"
          strokeWidth="1.5"
          strokeDasharray="2.2 2"
          className="stroke-zinc-500"
        />
        <path
          d="M7 4.25V7l1.75 1.25"
          fill="none"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="stroke-zinc-500"
        />
      </svg>
    );
  }

  if (columnId === TODO) {
    return (
      <svg viewBox="0 0 14 14" aria-hidden className={classes}>
        <circle
          cx="7"
          cy="7"
          r="6"
          fill="none"
          strokeWidth="1.5"
          className="stroke-zinc-400"
        />
      </svg>
    );
  }

  if (columnId === IN_PROGRESS) {
    return (
      <svg viewBox="0 0 14 14" aria-hidden className={classes}>
        <circle
          cx="7"
          cy="7"
          r="6"
          fill="none"
          strokeWidth="1.5"
          className="stroke-amber-500"
        />
        <path d="M7 3.5 A3.5 3.5 0 0 1 7 10.5 Z" className="fill-amber-500" />
      </svg>
    );
  }

  if (columnId === DONE) {
    return (
      <svg viewBox="0 0 14 14" aria-hidden className={classes}>
        <circle cx="7" cy="7" r="7" className="fill-violet-600" />
        <path
          d="M4.25 7.25 6.25 9.25 9.75 5.25"
          fill="none"
          stroke="white"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 14 14" aria-hidden className={classes}>
      <circle
        cx="7"
        cy="7"
        r="6"
        fill="none"
        strokeWidth="1.5"
        strokeDasharray="2.2 2"
        className="stroke-zinc-400"
      />
    </svg>
  );
}
