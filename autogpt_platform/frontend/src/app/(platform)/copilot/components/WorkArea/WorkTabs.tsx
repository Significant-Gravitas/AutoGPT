"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import {
  Add01Icon,
  ArrowLeftDoubleIcon,
  ArrowRightDoubleIcon,
  Cancel01Icon,
  CheckListIcon,
  File02Icon,
  Folder01Icon,
  Message01Icon,
  Sun03Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { useEffect, useRef, type KeyboardEvent } from "react";
import { useOverflowRight } from "@/app/(platform)/tasks/components/TaskBoard/useOverflowRight";

export interface TabView {
  key: string;
  title: string;
  icon: IconSvgElement;
}

interface Props {
  tabs: TabView[];
  activeKey: string | null;
  onSelect: (key: string) => void;
  onClose: (key: string) => void;
  /** Left out where there is no chat to open a new tab on. */
  onNew?: () => void;
  /** The sidebar is at its reading width, so the button offers to narrow it. */
  isWide: boolean;
  onToggleWidth: () => void;
}

export const TAB_ICONS = {
  task: CheckListIcon,
  artifact: File02Icon,
  context: Folder01Icon,
  start: Message01Icon,
  brief: Sun03Icon,
} as const;

/** The tabs across the sidebar's head: each with its own close, and a plus
 *  at the end for a new tab. More tabs first narrow, their names fading out,
 *  then the strip scrolls sideways. Left and Right move between them, Delete
 *  closes the one in focus. */
export function WorkTabs({
  tabs,
  activeKey,
  onSelect,
  onClose,
  onNew,
  isWide,
  onToggleWidth,
}: Props) {
  const { ref: stripRef, hiddenRight } = useOverflowRight<HTMLDivElement>();
  const buttons = useRef(new Map<string, HTMLButtonElement>());

  // A tab that comes forward scrolls into sight in the strip, never the page.
  useEffect(() => {
    if (!activeKey) return;
    buttons.current
      .get(activeKey)
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [activeKey, tabs.length]);

  function focusTab(key: string) {
    onSelect(key);
    buttons.current.get(key)?.focus({ preventScroll: true });
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = tabs.findIndex((tab) => tab.key === activeKey);
    if (index === -1) return;
    if (isKey(event, "Delete", "Backspace")) {
      event.preventDefault();
      onClose(tabs[index].key);
      return;
    }
    const next = isKey(event, "ArrowRight")
      ? tabs[(index + 1) % tabs.length]
      : isKey(event, "ArrowLeft")
        ? tabs[(index - 1 + tabs.length) % tabs.length]
        : isKey(event, "Home")
          ? tabs[0]
          : isKey(event, "End")
            ? tabs[tabs.length - 1]
            : null;
    if (!next) return;
    event.preventDefault();
    focusTab(next.key);
  }

  return (
    <div className="flex shrink-0 items-center">
      <div
        ref={stripRef}
        role="tablist"
        aria-label="Open tabs"
        onKeyDown={handleKeyDown}
        className={cn(
          "scrollbar-none flex min-w-0 items-center gap-0.5 overflow-x-auto overscroll-none p-0.5",
          // The end of the strip fades out only while more tabs wait behind it.
          hiddenRight && "mask-fade-right",
        )}
      >
        {tabs.map((tab) => {
          const isActive = tab.key === activeKey;
          return (
            <span
              key={tab.key}
              data-on={isActive || undefined}
              className={cn(
                "group/tab relative flex min-w-[5.75rem] shrink rounded-lg transition-colors duration-150 ease-out motion-reduce:transition-none",
                isActive
                  ? "bg-surface shadow-(--inbox-shadow-card)"
                  : "hover:bg-surface-wash",
              )}
            >
              <button
                ref={(element) => {
                  if (element) buttons.current.set(tab.key, element);
                  else buttons.current.delete(tab.key);
                }}
                type="button"
                role="tab"
                aria-selected={isActive}
                tabIndex={isActive ? 0 : -1}
                title={tab.title}
                onClick={() => onSelect(tab.key)}
                className={cn(
                  "inline-flex h-7 w-full max-w-[9.75rem] min-w-0 items-center gap-1.5 rounded-lg pr-7 pl-2.5 text-body-sm transition-colors duration-150 ease-out outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-inset motion-reduce:transition-none",
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
        })}
      </div>
      {onNew ? (
        <button
          type="button"
          aria-label="New tab"
          title="New tab"
          onClick={onNew}
          className="flex size-7 shrink-0 items-center justify-center rounded-lg text-foreground-text transition-colors duration-150 ease-out outline-none hover:bg-surface-wash hover:text-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-inset motion-reduce:transition-none"
        >
          <Icon icon={Add01Icon} size={14} tone="inherit" />
        </button>
      ) : null}
      <button
        type="button"
        aria-label={isWide ? "Make sidebar narrower" : "Make sidebar wider"}
        title={isWide ? "Make sidebar narrower" : "Make sidebar wider"}
        onClick={onToggleWidth}
        className="flex size-7 shrink-0 items-center justify-center rounded-lg text-foreground-text transition-colors duration-150 ease-out outline-none hover:bg-surface-wash hover:text-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-inset motion-reduce:transition-none"
      >
        <Icon
          icon={isWide ? ArrowRightDoubleIcon : ArrowLeftDoubleIcon}
          size={14}
          tone="inherit"
        />
      </button>
    </div>
  );
}
