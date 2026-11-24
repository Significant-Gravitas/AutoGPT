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
      className={cn(
        "relative overflow-hidden rounded-2xl-inner bg-surface",
        densityVariants({ density }),
      )}
    >
      <div
        aria-hidden
        data-slot="attention-card-featured-glow"
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the brand and indigo radial glow is a gradient, which has no token
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_60%_at_0%_0%,rgb(var(--color-brand)/0.10),transparent_60%),radial-gradient(120%_60%_at_100%_100%,rgb(var(--color-indigo)/0.06),transparent_60%)]"
      />
      <div data-slot="attention-card-featured-body" className="relative">
        {children}
      </div>
    </div>
  );
}
