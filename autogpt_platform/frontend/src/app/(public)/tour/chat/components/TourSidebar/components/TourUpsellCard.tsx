"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { GithubLogoIcon, SparkleIcon } from "@phosphor-icons/react";
import { buildTourPricingUrl, TOUR_GITHUB_URL } from "../../../constants";
import { trackTourCtaClick } from "../../../tracking";

// Only rendered while the demo is still playing — once it completes the
// sidebar hides this card and the end card in the chat carries the upsell.
export function TourUpsellCard() {
  return (
    <div className="relative flex flex-col rounded-xl border border-zinc-200/80 bg-white p-4 shadow-sm">
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
        Spin up agents like this in minutes, hosted for you or on your own
        infrastructure.
      </Text>
      <Button
        as="NextLink"
        href={buildTourPricingUrl("sidebar_card")}
        target="_blank"
        rel="noopener noreferrer"
        variant="primary"
        size="small"
        onClick={() =>
          trackTourCtaClick("pricing", { placement: "sidebar-card" })
        }
        className="mt-3 w-full shadow-[0_0_20px_-6px_rgba(124,58,237,0.6)] transition-shadow hover:shadow-[0_0_28px_-4px_rgba(124,58,237,0.75)]"
      >
        Start with Pro for $42.50/mo
      </Button>
      <Button
        as="NextLink"
        href={TOUR_GITHUB_URL}
        target="_blank"
        rel="noopener noreferrer"
        variant="ghost"
        size="small"
        onClick={() =>
          trackTourCtaClick("self-host", { placement: "sidebar-card" })
        }
        className="mt-1.5 w-full text-zinc-600"
        leftIcon={<GithubLogoIcon className="h-4 w-4" />}
      >
        Self-host free
      </Button>
    </div>
  );
}
