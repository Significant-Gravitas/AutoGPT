import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import type { BoardView } from "../useBoardView";

const LANES = 3;
const CARDS = 3;
const ROWS = 5;

interface Props {
  view: BoardView;
}

/** The first read, in the shape of what is coming: a few grey cards per
 *  lane on the board, a few rows in the list. */
export function BoardSkeleton({ view }: Props) {
  if (view === "list") {
    return (
      <div
        role="status"
        aria-busy
        aria-label="Loading tasks"
        className="grid gap-1 rounded-tray bg-surface-tray p-1.5"
      >
        <Skeleton className="m-1.5 h-5 w-24" />
        {Array.from({ length: ROWS }, (_, index) => (
          <Skeleton key={index} className="h-11" />
        ))}
      </div>
    );
  }
  return (
    <div
      role="status"
      aria-busy
      aria-label="Loading tasks"
      className="flex min-h-56 flex-1 gap-2 overflow-hidden"
    >
      {Array.from({ length: LANES }, (_, lane) => (
        <div
          key={lane}
          className="flex min-w-64 flex-1 basis-0 flex-col gap-1.5 rounded-tray bg-surface-tray p-1.5"
        >
          <Skeleton className="m-2 h-5 w-24" />
          {Array.from({ length: CARDS }, (_, card) => (
            <Skeleton key={card} className="h-24" />
          ))}
        </div>
      ))}
    </div>
  );
}
