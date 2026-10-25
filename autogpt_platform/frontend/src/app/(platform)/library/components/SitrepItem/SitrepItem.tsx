"use client";

import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Text } from "@/components/atoms/Text/Text";
import NextLink from "next/link";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import type { SitrepItemData, SitrepPriority } from "../../types";
import { ContextualActionButton } from "../ContextualActionButton/ContextualActionButton";
import { buildAutoPilotPrompt } from "./helpers";
import {
  Alert02Icon,
  AlertCircleIcon,
  Calendar03Icon,
  Chatting01Icon,
  CheckmarkCircle02Icon,
  ClockFadingIcon,
  EarIcon,
  EyeIcon,
  MoonIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useImageFallback } from "@/hooks/useImageFallback";

interface Props {
  item: SitrepItemData;
}

const PRIORITY_CONFIG: Record<
  SitrepPriority,
  {
    icon?: IconSvgElement;
    color: string;
    bg: string;
    cssSpinner?: boolean;
  }
> = {
  error: {
    icon: AlertCircleIcon,
    color: "text-error",
    bg: "bg-error-surface",
  },
  running: {
    color: "text-foreground-emphasis",
    bg: "",
    cssSpinner: true,
  },
  stale: {
    icon: ClockFadingIcon,
    color: "text-yellow-600",
    bg: "bg-yellow-50",
  },
  warning: {
    icon: Alert02Icon,
    color: "text-amber-600",
    bg: "bg-warning-surface",
  },
  success: {
    icon: CheckmarkCircle02Icon,
    color: "text-success",
    bg: "bg-success-surface",
  },
  listening: {
    icon: EarIcon,
    color: "text-brand",
    bg: "bg-purple-50",
  },
  scheduled: {
    icon: Calendar03Icon,
    color: "text-yellow-600",
    bg: "bg-yellow-50",
  },
  idle: {
    icon: MoonIcon,
    color: "text-foreground-subtle",
    bg: "bg-surface-hover",
  },
};

export function SitrepItem({ item }: Props) {
  const config = PRIORITY_CONFIG[item.priority];
  const { showImage, handleImageError } = useImageFallback(item.agentImageUrl);
  const router = useRouter();

  function handleAskAutoPilot() {
    const prompt = buildAutoPilotPrompt(item);
    const encoded = encodeURIComponent(prompt);
    router.push(`/home?autosubmit=true#prompt=${encoded}`);
  }

  return (
    <div
      className={cn(
        "flex flex-col gap-2 rounded-medium border border-border-base/50 bg-transparent p-2 sm:flex-row sm:items-center sm:gap-3",
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        {showImage && item.agentImageUrl ? (
          <img
            src={item.agentImageUrl}
            alt={item.agentName}
            className="h-6 w-6 flex-shrink-0 rounded-full object-cover"
            onError={handleImageError}
          />
        ) : (
          <div
            className={cn(
              "flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full",
              config.bg,
            )}
          >
            {config.cssSpinner ? (
              <div
                // TODO(tokens): no token for the ring spinner's gradient and mask (#0000, #000)
                className="aspect-square h-[21px] w-[21px] animate-spin rounded-[50%] text-foreground-emphasis [-webkit-mask:radial-gradient(farthest-side,#0000_calc(100%_-_3px),#000_0)] [background:radial-gradient(farthest-side,currentColor_94%,#0000)_top/3px_3px_no-repeat,conic-gradient(#0000_30%,currentColor)] [mask:radial-gradient(farthest-side,#0000_calc(100%_-_3px),#000_0)]"
              />
            ) : (
              config.icon && (
                <Icon icon={config.icon} size={14} className={config.color} />
              )
            )}
          </div>
        )}

        <div className="min-w-0 flex-1">
          <Text
            variant="body-medium"
            className="leading-tight text-foreground-strong"
          >
            {item.agentName}
          </Text>
          <OverflowText
            value={item.message}
            variant="small"
            className="leading-tight text-foreground-tertiary"
          />
        </div>
      </div>

      <div className="flex flex-shrink-0 flex-wrap items-center justify-center gap-1.5 sm:flex-nowrap sm:justify-end">
        {item.priority === "success" || item.priority === "warning" ? (
          <NextLink
            href={`/library/agents/${item.agentID}${item.executionID ? `?activeItem=${item.executionID}` : ""}`}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-body-sm font-medium leading-inherit text-foreground-secondary transition-colors hover:bg-surface-muted hover:text-foreground-emphasis"
          >
            <Icon icon={EyeIcon} size={14} className="shrink-0" />
            See task
          </NextLink>
        ) : (
          <ContextualActionButton
            status={item.status}
            agentID={item.agentID}
            executionID={item.executionID}
          />
        )}
        <button
          type="button"
          onClick={handleAskAutoPilot}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-body-sm font-medium leading-inherit text-foreground-secondary transition-colors hover:bg-surface-muted hover:text-foreground-emphasis"
        >
          <Icon icon={Chatting01Icon} size={14} className="shrink-0" />
          Ask Otto
        </button>
      </div>
    </div>
  );
}
