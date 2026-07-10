"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { GithubLogoIcon, SparkleIcon } from "@phosphor-icons/react";

const PRICING_URL = "https://agpt.co/pricing";
const GITHUB_URL = "https://github.com/Significant-Gravitas/AutoGPT";

export function TourUpsellCard() {
  return (
    <div className="flex flex-col rounded-xl border border-zinc-200/80 bg-white p-4 shadow-[0_1px_2px_rgba(0,0,0,0.03),0_12px_32px_-16px_rgba(99,102,241,0.45)]">
      <div className="flex items-center gap-1.5">
        <SparkleIcon
          size={14}
          weight="fill"
          className="shrink-0 text-violet-600"
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
        className="mt-3 w-full"
      >
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
  );
}
