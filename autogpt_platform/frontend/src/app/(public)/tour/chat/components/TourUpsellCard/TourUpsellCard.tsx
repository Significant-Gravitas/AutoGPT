"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowCounterClockwiseIcon, SparkleIcon } from "@phosphor-icons/react";

interface Props {
  onReplay: () => void;
}

export function TourUpsellCard({ onReplay }: Props) {
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-zinc-200 bg-white p-5 shadow-[0_2px_8px_rgba(0,0,0,0.04),0_0_32px_-4px_rgba(99,102,241,0.4)]">
      <div className="flex items-start gap-3">
        <SparkleIcon
          size={22}
          weight="fill"
          className="mt-0.5 shrink-0 text-violet-600"
        />
        <div className="flex flex-col gap-1">
          <Text variant="large-medium" className="text-zinc-900">
            Ready to build your own?
          </Text>
          <Text variant="body" className="text-zinc-600">
            Spin up your own AI agents in minutes. Sign up free to start
            building, no credit card required.
          </Text>
        </div>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row sm:justify-end">
        <Button
          variant="ghost"
          size="small"
          leftIcon={<ArrowCounterClockwiseIcon className="h-4 w-4" />}
          onClick={onReplay}
        >
          Replay demo
        </Button>
        <Button
          as="NextLink"
          href="https://agpt.co/pricing"
          target="_blank"
          rel="noopener noreferrer"
          variant="secondary"
          size="small"
        >
          See pricing
        </Button>
        <Button as="NextLink" href="/signup" variant="primary" size="small">
          Sign up free
        </Button>
      </div>
    </div>
  );
}
