import { cn } from "@/lib/utils";

interface Props {
  passage: string;
  // Off where the passage can be released from the row: what is released must be shown whole.
  clamp?: boolean;
}

export function HeldPassageQuote({ passage, clamp = true }: Props) {
  return (
    <blockquote
      aria-label="What it says"
      className={cn(
        "mt-1 border-l-2 border-amber-400 pl-2 text-sm text-zinc-600 [overflow-wrap:anywhere]",
        clamp && "line-clamp-2",
      )}
    >
      {passage}
    </blockquote>
  );
}
