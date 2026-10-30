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
        "mt-1 border-l-2 border-warning pl-2 text-sm [overflow-wrap:anywhere] text-foreground-secondary",
        clamp && "line-clamp-2",
      )}
    >
      {passage}
    </blockquote>
  );
}
