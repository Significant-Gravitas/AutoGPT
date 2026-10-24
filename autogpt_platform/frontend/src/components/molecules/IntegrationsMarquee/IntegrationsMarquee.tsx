"use client";

import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";

type Variant = "light" | "dark";

interface Props {
  variant?: Variant;
  className?: string;
}

const ROW_A = ["github", "google", "notion", "airtable", "openai", "linear"];
const ROW_B = [
  "discord",
  "anthropic",
  "hubspot",
  "reddit",
  "telegram",
  "ideogram",
];

export function IntegrationsMarquee({ variant = "light", className }: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <div
      aria-hidden
      data-slot="integrations-marquee"
      className={cn(
        "relative flex h-[200px] w-[340px] flex-col justify-center gap-3 overflow-hidden [-webkit-mask-image:linear-gradient(to_right,transparent_0%,black_18%,black_82%,transparent_100%)] [mask-image:linear-gradient(to_right,transparent_0%,black_18%,black_82%,transparent_100%)]",
        className,
      )}
    >
      <MarqueeRow
        providers={ROW_A}
        direction="left"
        reduceMotion={!!reduceMotion}
        variant={variant}
      />
      <MarqueeRow
        providers={ROW_B}
        direction="right"
        reduceMotion={!!reduceMotion}
        variant={variant}
      />
    </div>
  );
}

function MarqueeRow({
  providers,
  direction,
  reduceMotion,
  variant,
}: {
  providers: string[];
  direction: "left" | "right";
  reduceMotion: boolean;
  variant: Variant;
}) {
  const animateX = direction === "left" ? ["0%", "-50%"] : ["-50%", "0%"];

  return (
    <motion.div
      data-slot="integrations-marquee-row"
      className="flex w-max gap-3 will-change-transform"
      animate={reduceMotion ? undefined : { x: animateX }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 22, ease: "linear", repeat: Infinity }
      }
    >
      {[...providers, ...providers].map((id, i) => (
        <GhostIntegrationCard key={`${id}-${i}`} id={id} variant={variant} />
      ))}
    </motion.div>
  );
}

function GhostIntegrationCard({
  id,
  variant,
}: {
  id: string;
  variant: Variant;
}) {
  const isDark = variant === "dark";
  return (
    <div
      // TODO(tokens): no shadow token for 0 1px 2px rgba(0,0,0,0.03).
      data-slot="integrations-marquee-card"
      className={cn(
        "flex h-[58px] w-[180px] shrink-0 items-center gap-3 rounded-xl border px-3 shadow-[0_1px_2px_rgba(0,0,0,0.03)]",
        isDark
          ? "border-surface/10 bg-surface/[0.04]"
          : "border-border-base/50 bg-surface/50 opacity-70",
      )}
    >
      <div
        data-slot="integrations-marquee-card-logo-wrap"
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-lg",
          isDark ? "bg-surface/[0.06]" : "bg-surface-muted/60",
        )}
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- decorative tiny logo, no LCP candidate */}
        <img
          src={`/integrations/${id}.png`}
          alt=""
          width={20}
          height={20}
          loading="lazy"
          decoding="async"
          data-slot="integrations-marquee-card-logo"
          className={cn(
            "size-5 rounded-sm object-contain",
            isDark ? "" : "opacity-80",
          )}
        />
      </div>
      <div
        data-slot="integrations-marquee-card-lines"
        className="flex flex-1 flex-col gap-1.5"
      >
        <div
          data-slot="integrations-marquee-card-line"
          className={cn(
            "h-2 w-3/4 rounded-full",
            isDark ? "bg-surface/15" : "bg-surface-hover/80",
          )}
        />
        <div
          data-slot="integrations-marquee-card-line"
          className={cn(
            "h-2 w-1/2 rounded-full",
            isDark ? "bg-surface/10" : "bg-surface-hover/80",
          )}
        />
      </div>
    </div>
  );
}
