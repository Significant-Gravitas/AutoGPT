"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowCounterClockwiseIcon,
  GithubLogoIcon,
  SparkleIcon,
} from "@phosphor-icons/react";

interface Props {
  onReplay: () => void;
}

export function TourUpsellBanner({ onReplay }: Props) {
  return (
    <div className="fixed inset-x-0 bottom-0 z-50 border-t border-zinc-200 bg-white shadow-[0_-2px_8px_rgba(0,0,0,0.04),0_0_32px_-4px_rgba(99,102,241,0.4)] duration-500 animate-in fade-in slide-in-from-bottom-2">
      <div className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:gap-4 sm:px-6">
        <div className="flex items-start gap-3">
          <SparkleIcon
            size={22}
            weight="fill"
            className="mt-0.5 shrink-0 text-violet-600"
          />
          <div className="flex flex-col">
            <Text variant="large-medium" className="text-zinc-900">
              Ready to build your own?
            </Text>
            <Text variant="body" className="text-zinc-600">
              Spin up agents like this in minutes — hosted for you, or on your
              own infrastructure.
            </Text>
          </div>
        </div>

        <div className="flex flex-col gap-2 sm:ml-auto sm:flex-row sm:items-center sm:gap-3">
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
            href="https://github.com/Significant-Gravitas/AutoGPT"
            target="_blank"
            rel="noopener noreferrer"
            variant="secondary"
            size="small"
            leftIcon={<GithubLogoIcon className="h-4 w-4" />}
          >
            Self-host free
          </Button>
          <Button
            as="NextLink"
            href="https://agpt.co/pricing"
            target="_blank"
            rel="noopener noreferrer"
            variant="primary"
            size="small"
            className="shadow-[0_0_24px_-4px_rgba(124,58,237,0.7)] transition-shadow hover:shadow-[0_0_32px_-2px_rgba(124,58,237,0.85)]"
          >
            Start with Pro — $42.50/mo
          </Button>
        </div>
      </div>
    </div>
  );
}
