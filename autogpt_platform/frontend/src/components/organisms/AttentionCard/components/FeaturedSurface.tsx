import type { VariantProps } from "class-variance-authority";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { densityVariants } from "../helpers";

interface FeaturedSurfaceProps extends VariantProps<typeof densityVariants> {
  children: ReactNode;
}

// The featured card draws its 1px gradient border as an outer fill, so its
// surface, padding and glow sit one element in.
export function FeaturedSurface({ density, children }: FeaturedSurfaceProps) {
  return (
    <div
      data-slot="attention-card-featured-surface"
      // TODO(tokens): no radius token for 15px, the 16px card inside its 1px
      // border.
      className={cn(
        "relative overflow-hidden rounded-[15px] bg-surface",
        densityVariants({ density }),
      )}
    >
      <div
        aria-hidden
        data-slot="attention-card-featured-glow"
        // TODO(tokens): no token for the purple and indigo radial glow.
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_60%_at_0%_0%,rgba(168,85,247,0.10),transparent_60%),radial-gradient(120%_60%_at_100%_100%,rgba(99,102,241,0.06),transparent_60%)]"
      />
      <div data-slot="attention-card-featured-body" className="relative">
        {children}
      </div>
    </div>
  );
}
