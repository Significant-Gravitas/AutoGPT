"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { GithubLogoIcon, SparkleIcon } from "@phosphor-icons/react";
import { useTourStore } from "../../../tourStore";

const PRICING_URL = "https://agpt.co/pricing";
const GITHUB_URL = "https://github.com/Significant-Gravitas/AutoGPT";

export function TourUpsellCard() {
  // The glow and moving effects (spinning border, shine sweep, pulsing
  // sparkle) would fight the streaming transcript for attention, so the card
  // sits quiet and only lights up once the demo has played through.
  const isDemoComplete = useTourStore((s) => s.isDemoComplete);

  return (
    // Border: a p-px shell whose backdrop is a plain hairline while the demo
    // runs and an oversized spinning conic gradient once it completes (same
    // trick as PlanCard's badge).
    <div
      className={cn(
        "relative rounded-xl p-px",
        isDemoComplete
          ? "shadow-[0_0_20px_-4px_rgba(139,92,246,0.45),0_0_44px_-8px_rgba(139,92,246,0.4)]"
          : "shadow-sm",
      )}
    >
      <span
        aria-hidden="true"
        className="absolute inset-0 overflow-hidden rounded-xl"
      >
        <span
          className={cn(
            "absolute -inset-[150%]",
            isDemoComplete
              ? "animate-[spin_6s_linear_infinite] bg-[conic-gradient(from_0deg,#ede9fe,#c4b5fd,#8b5cf6,#ede9fe,#ddd6fe,#a78bfa,#ede9fe)]"
              : "bg-zinc-200/80",
          )}
        />
      </span>

      <div className="relative flex flex-col rounded-[11px] bg-white p-4">
        <div className="flex items-center gap-1.5">
          <SparkleIcon
            size={14}
            weight="fill"
            className={cn(
              "shrink-0 text-violet-600",
              isDemoComplete && "animate-pulse",
            )}
          />
          <Text variant="body-medium" className="text-zinc-900">
            Ready to build your own?
          </Text>
        </div>
        <Text variant="small" className="mt-1 text-zinc-500">
          Spin up agents like this in minutes, hosted for you or on your own
          infrastructure.
        </Text>
        <Button
          as="NextLink"
          href={PRICING_URL}
          target="_blank"
          rel="noopener noreferrer"
          variant="primary"
          size="small"
          className="relative mt-3 w-full overflow-hidden shadow-[0_0_20px_-6px_rgba(124,58,237,0.6)] transition-shadow hover:shadow-[0_0_28px_-4px_rgba(124,58,237,0.75)]"
        >
          {isDemoComplete && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-y-0 left-0 w-1/3 -skew-x-12 animate-[progress-bar_2.6s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/40 to-transparent"
            />
          )}
          Start with Pro for $42.50/mo
        </Button>
        <Button
          as="NextLink"
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          variant="ghost"
          size="small"
          className="mt-1.5 w-full text-zinc-600"
          leftIcon={<GithubLogoIcon className="h-4 w-4" />}
        >
          Self-host free
        </Button>
      </div>
    </div>
  );
}
