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
      className={cn(
        "relative flex h-[200px] w-[340px] flex-col justify-center gap-3 overflow-hidden",
        className,
      )}
      style={{
        maskImage:
          "linear-gradient(to right, transparent 0%, black 18%, black 82%, transparent 100%)",
        WebkitMaskImage:
          "linear-gradient(to right, transparent 0%, black 18%, black 82%, transparent 100%)",
      }}
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
      className={cn(
        "flex h-[58px] w-[180px] shrink-0 items-center gap-3 rounded-xl border px-3 shadow-[0_1px_2px_rgba(0,0,0,0.03)]",
        isDark
          ? "border-white/10 bg-white/[0.04]"
          : "border-zinc-200/50 bg-white/50 opacity-70",
      )}
    >
      <div
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-lg",
          isDark ? "bg-white/[0.06]" : "bg-zinc-50/60",
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
          className={cn(
            "size-5 rounded-sm object-contain",
            isDark ? "" : "opacity-80",
          )}
        />
      </div>
      <div className="flex flex-1 flex-col gap-1.5">
        <div
          className={cn(
            "h-2 w-3/4 rounded-full",
            isDark ? "bg-white/15" : "bg-zinc-100/80",
          )}
        />
        <div
          className={cn(
            "h-2 w-1/2 rounded-full",
            isDark ? "bg-white/10" : "bg-zinc-100/80",
          )}
        />
      </div>
    </div>
  );
}
