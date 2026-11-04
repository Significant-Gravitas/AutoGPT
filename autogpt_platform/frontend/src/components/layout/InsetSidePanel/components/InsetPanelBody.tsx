import { useScrollEdges } from "@/lib/hooks/useScrollEdges";
import { cn } from "@/lib/utils";
import { ReactNode, useRef } from "react";

const EDGE_FADE_CLASS =
  "pointer-events-none absolute inset-x-0 z-10 h-8 from-sidebar via-sidebar/70 to-transparent transition-opacity duration-200";

interface Props {
  children: ReactNode;
}

export function InsetPanelBody({ children }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { hiddenAbove, hiddenBelow } = useScrollEdges(scrollRef);

  return (
    <div className="relative -mx-0.5 mt-3 flex min-h-0 flex-1 flex-col">
      <div
        aria-hidden
        className={cn(
          EDGE_FADE_CLASS,
          "top-0 bg-gradient-to-b",
          hiddenAbove ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        ref={scrollRef}
        className="scrollbar-none min-h-0 flex-1 overflow-y-auto overscroll-contain px-0.5 pb-1"
      >
        <div>{children}</div>
      </div>
      <div
        aria-hidden
        className={cn(
          EDGE_FADE_CLASS,
          "bottom-0 bg-gradient-to-t",
          hiddenBelow ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}
