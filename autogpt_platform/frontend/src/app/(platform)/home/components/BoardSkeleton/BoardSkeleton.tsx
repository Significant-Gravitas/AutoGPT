import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

const CARDS_PER_COLUMN = [3, 2, 1];

export function BoardSkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading board"
      className="flex min-h-0 flex-1 gap-3 overflow-hidden"
    >
      {CARDS_PER_COLUMN.map((count, column) => (
        <div
          key={column}
          className="flex w-72 shrink-0 flex-col gap-2 self-start rounded-xl bg-zinc-100/80 p-2"
        >
          <Skeleton className="m-1 h-4 w-24 bg-zinc-200/70" />
          {Array.from({ length: count }, (_, card) => (
            <Skeleton key={card} className="h-[4.75rem] w-full bg-white" />
          ))}
        </div>
      ))}
    </div>
  );
}
