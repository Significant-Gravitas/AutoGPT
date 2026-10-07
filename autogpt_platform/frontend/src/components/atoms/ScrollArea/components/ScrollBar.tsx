import { cn } from "@/lib/utils";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";

interface Props {
  orientation: "vertical" | "horizontal";
}

// Radix hides the native scrollbar, so this mirrors scrollbarStyles
// (styles/scrollbars.ts): a thin zinc-300 thumb on a transparent track.
export function ScrollBar({ orientation }: Props) {
  return (
    <ScrollAreaPrimitive.ScrollAreaScrollbar
      orientation={orientation}
      className={cn(
        "flex touch-none select-none p-px transition-colors",
        orientation === "vertical" && "h-full w-2.5",
        orientation === "horizontal" && "h-2.5 flex-col",
      )}
    >
      <ScrollAreaPrimitive.ScrollAreaThumb className="relative flex-1 rounded-full bg-zinc-300 hover:bg-zinc-400" />
    </ScrollAreaPrimitive.ScrollAreaScrollbar>
  );
}
