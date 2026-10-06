"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowRight02Icon,
  GithubIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { TOUR_GITHUB_URL } from "../../../constants";
import { trackTourCtaClick } from "../../../tracking";

export function TourUpsellCard() {
  return (
    <Card className="flex flex-col gap-4 border border-zinc-200 p-4 shadow-subtle">
      <div className="flex items-center gap-2.5">
        <span className="flex size-9 items-center justify-center rounded-medium bg-purple-100 text-purple-700">
          <Icon icon={SparklesIcon} size={18} aria-hidden />
        </span>
        <Text variant="eyebrow" tone="secondary">
          Free trial
        </Text>
      </div>
      <div className="space-y-1.5">
        <Text variant="h5" as="h2" tone="primary">
          Your AI team starts here
        </Text>
        <Text variant="body" tone="secondary">
          Build agents and put AI experts to work on your everyday tasks.
        </Text>
      </div>
      <div className="flex flex-col gap-1">
        <Button
          as="NextLink"
          href="/signup"
          variant="primary"
          size="small"
          onClick={() =>
            trackTourCtaClick("free-trial", { placement: "sidebar-card" })
          }
          rightIcon={<Icon icon={ArrowRight02Icon} size={16} aria-hidden />}
          className="w-full"
        >
          Start free trial
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
          leftIcon={<Icon icon={GithubIcon} size={14} aria-hidden />}
          className="w-full text-xs text-zinc-600"
        >
          Self-host instead
        </Button>
      </div>
    </Card>
  );
}
