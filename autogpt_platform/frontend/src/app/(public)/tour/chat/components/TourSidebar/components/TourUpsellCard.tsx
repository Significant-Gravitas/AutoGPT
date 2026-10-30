"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { environment } from "@/services/environment";
import {
  ArrowRight02Icon,
  GithubIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { TOUR_GITHUB_URL } from "../../../constants";
import { trackTourCtaClick } from "../../../tracking";

type Props = {
  surface: "tour" | "marketplace";
};

export function TourUpsellCard({ surface }: Props) {
  const isCloud = environment.isCloud();

  function handleSignupClick() {
    trackTourCtaClick(isCloud ? "free-trial" : "signup", {
      placement: "sidebar-card",
      surface,
    });
  }

  function handleSelfHostClick() {
    trackTourCtaClick("self-host", { placement: "sidebar-card", surface });
  }

  return (
    <div className="flex flex-col gap-4 rounded-large border border-border-base bg-surface p-4 shadow-subtle">
      <div className="flex items-center gap-2.5">
        <span className="flex size-9 items-center justify-center rounded-medium bg-accent-surface text-accent-strong">
          <Icon icon={SparklesIcon} size={18} aria-hidden />
        </span>
        <Text variant="eyebrow" tone="secondary">
          {isCloud ? "Free trial" : "Get started"}
        </Text>
      </div>
      <div className="space-y-1.5">
        <Text variant="h5" as="h2" tone="primary">
          Your AI team starts here
        </Text>
        <Text variant="body" tone="secondary">
          Build workflows and put AI experts to work on your everyday tasks.
        </Text>
      </div>
      <div className="flex flex-col gap-1">
        <Button
          as="NextLink"
          href="/signup"
          variant="primary"
          size="small"
          onClick={handleSignupClick}
          rightIcon={<Icon icon={ArrowRight02Icon} size={16} aria-hidden />}
          className="w-full"
        >
          {isCloud ? "Start free trial" : "Create account"}
        </Button>
        <Button
          as="NextLink"
          href={TOUR_GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          variant="ghost"
          size="small"
          onClick={handleSelfHostClick}
          leftIcon={<Icon icon={GithubIcon} size={14} aria-hidden />}
          // oxlint-disable-next-line design-system/no-restyle -- quiet 12px secondary link; no Button variant has small muted text at this height
          className="w-full text-xs text-foreground-secondary"
        >
          Self-host instead
        </Button>
      </div>
    </div>
  );
}
