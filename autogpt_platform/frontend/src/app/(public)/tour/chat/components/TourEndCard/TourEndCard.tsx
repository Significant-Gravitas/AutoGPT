"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CheckIcon, PlayIcon } from "@phosphor-icons/react";
import {
  buildTourPricingUrl,
  TOUR_DEMO_CLAIM_SECONDS,
  TOUR_GITHUB_URL,
} from "../../constants";
import { useTourEndCard } from "./useTourEndCard";

export function TourEndCard() {
  const { handlePricingClick, handleSelfHostClick, handleWatchAnother } =
    useTourEndCard();

  return (
    <div className="mx-auto w-full max-w-md rounded-2xl border border-zinc-200/70 bg-white p-5 shadow-sm duration-500 animate-in fade-in slide-in-from-bottom-2 sm:p-6">
      <div className="flex flex-col items-center gap-2 text-center">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1">
          <CheckIcon className="size-3.5 text-emerald-700" weight="bold" />
          <Text variant="small-medium" className="!text-emerald-700">
            Agent built &amp; first run complete — {TOUR_DEMO_CLAIM_SECONDS}{" "}
            seconds
          </Text>
        </span>
        <Text variant="h4" as="h3" className="mt-1 text-zinc-900">
          That took {TOUR_DEMO_CLAIM_SECONDS} seconds.
          <br />
          Yours will too.
        </Text>
        <Text variant="body" className="text-zinc-500">
          Same chat, your real tasks — hosted and running for you.
        </Text>
      </div>

      <div className="mt-5 flex flex-col gap-3">
        <Button
          as="NextLink"
          href={buildTourPricingUrl("end_card")}
          target="_blank"
          rel="noopener noreferrer"
          variant="primary"
          onClick={handlePricingClick}
          className="h-auto w-full flex-col gap-0 py-2.5 shadow-[0_0_20px_-6px_rgba(124,58,237,0.6)]"
        >
          <span className="font-medium">Make this agent yours</span>
          <span className="text-xs font-normal opacity-80">
            Start with Pro · $42.50/mo · cancel anytime
          </span>
        </Button>
        <Button
          variant="secondary"
          onClick={handleWatchAnother}
          leftIcon={<PlayIcon className="size-4" weight="fill" />}
          className="w-full"
        >
          Watch another scenario
        </Button>
        <a
          href={TOUR_GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          onClick={handleSelfHostClick}
          className="mt-1 text-center text-sm text-zinc-500 underline underline-offset-2 hover:text-zinc-700"
        >
          or self-host free
        </a>
      </div>
    </div>
  );
}
