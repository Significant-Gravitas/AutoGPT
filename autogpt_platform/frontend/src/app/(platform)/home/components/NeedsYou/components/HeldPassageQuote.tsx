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
        // TODO(tokens): no exact role (border-amber-400 #FBBF24)
        "mt-1 border-l-2 border-amber-400 pl-2 text-sm text-foreground-secondary [overflow-wrap:anywhere]",
        clamp && "line-clamp-2",
      )}
    >
      {passage}
    </blockquote>
  );
}
