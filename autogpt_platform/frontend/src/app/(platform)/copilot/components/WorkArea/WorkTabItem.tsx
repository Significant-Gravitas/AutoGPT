"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { TabView } from "./WorkTabs";

interface Props {
  tab: TabView;
  isActive: boolean;
  buttonRef: (element: HTMLButtonElement | null) => void;
  onSelect: (key: string) => void;
  onClose: (key: string) => void;
}

/** One tab of the sidebar's strip. A long name fades out at the tab's end;
 *  pointing at the tab or moving focus onto it shows the whole name. */
export function WorkTabItem({
  tab,
  isActive,
  buttonRef,
  onSelect,
  onClose,
}: Props) {
  return (
    <span
      data-on={isActive || undefined}
      className={cn(
        "group/tab relative flex min-w-[5.75rem] shrink rounded-lg transition-colors duration-150 ease-out motion-reduce:transition-none",
        isActive
          ? "bg-surface shadow-(--inbox-shadow-card)"
          : "hover:bg-surface-wash",
      )}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            ref={buttonRef}
            type="button"
            role="tab"
            aria-selected={isActive}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onSelect(tab.key)}
            className={cn(
              "inline-flex h-7 w-full max-w-[15rem] min-w-0 items-center gap-1.5 rounded-lg pr-7 pl-2.5 text-body-sm transition-colors duration-150 ease-out outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-inset motion-reduce:transition-none",
              isActive
                ? "text-foreground-ink"
                : "text-foreground-text group-hover/tab:text-foreground-ink",
            )}
          >
            <Icon
              icon={tab.icon}
              size={14}
              tone="inherit"
              className="shrink-0"
            />
            <span className="min-w-0 overflow-hidden mask-fade-right pr-2 whitespace-nowrap">
              {tab.title}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-80 break-words">
          {tab.title}
        </TooltipContent>
      </Tooltip>
      <button
        type="button"
        tabIndex={-1}
        aria-label={`Close ${tab.title}`}
        onClick={() => onClose(tab.key)}
        className={cn(
          "absolute top-1.5 right-1 flex size-4 items-center justify-center rounded-sm text-foreground-text transition-opacity duration-100 ease-out outline-none hover:bg-surface-wash hover:text-foreground-ink",
          isActive
            ? "opacity-100"
            : "opacity-0 group-focus-within/tab:opacity-100 group-hover/tab:opacity-100",
        )}
      >
        <Icon icon={Cancel01Icon} size={10} tone="inherit" />
      </button>
    </span>
  );
}
