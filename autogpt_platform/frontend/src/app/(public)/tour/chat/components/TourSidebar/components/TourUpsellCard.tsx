"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { GithubLogoIcon, SparkleIcon } from "@phosphor-icons/react";

const PRICING_URL = "https://agpt.co/pricing";
const GITHUB_URL = "https://github.com/Significant-Gravitas/AutoGPT";

export function TourUpsellCard() {
  return (
    // Animated gradient border: a p-px shell with an oversized spinning conic
    // gradient behind the white inner card (same trick as PlanCard's badge).
    <div className="relative rounded-xl p-px shadow-[0_0_20px_-4px_rgba(139,92,246,0.45),0_0_44px_-8px_rgba(139,92,246,0.4)]">
      <span
        aria-hidden="true"
        className="absolute inset-0 overflow-hidden rounded-xl"
      >
        <span className="absolute -inset-[150%] animate-[spin_6s_linear_infinite] bg-[conic-gradient(from_0deg,#ede9fe,#c4b5fd,#8b5cf6,#ede9fe,#ddd6fe,#a78bfa,#ede9fe)]" />
      </span>

      <div className="relative flex flex-col rounded-[11px] bg-white p-4">
        <div className="flex items-center gap-1.5">
          <SparkleIcon
            size={14}
            weight="fill"
            className="shrink-0 animate-pulse text-violet-600"
          />
          <Text variant="body-medium" className="text-zinc-900">
            Ready to build your own?
          </Text>
        </div>
        <Text variant="small" className="mt-1 text-zinc-500">
          Spin up agents like this in minutes — hosted for you, or on your own
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
          {/* Shine sweep gliding across the CTA every few seconds. */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 left-0 w-1/3 -skew-x-12 animate-[progress-bar_2.6s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/40 to-transparent"
          />
          Start with Pro — $42.50/mo
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
