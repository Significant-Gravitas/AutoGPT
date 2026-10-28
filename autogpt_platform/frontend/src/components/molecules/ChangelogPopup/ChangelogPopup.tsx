"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowUpRight01Icon,
  Cancel01Icon,
  NewsIcon,
} from "@hugeicons/core-free-icons";
import { useChangelog } from "./useChangelog";

export function ChangelogPopup() {
  const {
    isVisible,
    isFading,
    latestEntry,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
  } = useChangelog();

  if (!isVisible || !latestEntry) return null;

  const highlights = latestEntry.highlights
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3)
    .map((item) => item.charAt(0).toUpperCase() + item.slice(1));

  return (
    <div
      data-slot="changelog-popup"
      className={`fixed bottom-4 right-4 z-50 w-[360px] max-w-[calc(100vw-2rem)] transition-all duration-500 ease-out sm:bottom-6 sm:right-6 ${
        isFading
          ? "pointer-events-none translate-y-2 opacity-0"
          : "translate-y-0 opacity-100"
      }`}
      onPointerEnter={(event) => {
        if (event.pointerType !== "touch") pauseAutoDismiss("hover");
      }}
      onPointerLeave={(event) => {
        if (event.pointerType !== "touch") resumeAutoDismiss("hover");
      }}
      onPointerCancel={(event) => {
        if (event.pointerType !== "touch") resumeAutoDismiss("hover");
      }}
      onFocus={() => pauseAutoDismiss("focus")}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          resumeAutoDismiss("focus");
        }
      }}
      role="dialog"
      aria-label="What's new"
    >
      <div
        data-slot="changelog-popup-card"
        className="rounded-2xl border border-border bg-background p-5 shadow-lg"
      >
        <div
          data-slot="changelog-popup-header"
          className="flex items-start gap-3"
        >
          <div
            data-slot="changelog-popup-badge"
            className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-secondary text-foreground"
          >
            <Icon
              data-slot="changelog-popup-badge-icon"
              icon={NewsIcon}
              size={18}
            />
          </div>
          <div data-slot="changelog-popup-heading" className="min-w-0 flex-1">
            <Text
              variant="body-medium"
              data-slot="changelog-popup-title"
              className="text-foreground"
            >
              What&apos;s new
            </Text>
            <Text
              variant="small"
              data-slot="changelog-popup-date"
              className="text-muted-foreground"
            >
              {latestEntry.dateRange}
            </Text>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            withTooltip={false}
            onClick={dismiss}
            data-slot="changelog-popup-dismiss"
            className="-mr-1 -mt-1 text-muted-foreground hover:text-foreground"
            aria-label="Dismiss changelog"
          >
            <Icon
              data-slot="changelog-popup-dismiss-icon"
              icon={Cancel01Icon}
              size={16}
            />
          </Button>
        </div>

        <ul data-slot="changelog-popup-highlights" className="my-4 space-y-1.5">
          {highlights.map((item) => (
            <li
              key={item}
              data-slot="changelog-popup-highlight"
              className="flex items-start gap-2.5"
            >
              <span
                aria-hidden="true"
                data-slot="changelog-popup-bullet"
                className="mt-2.5 size-1 shrink-0 rounded-full bg-muted-foreground"
              />
              <Text
                variant="body"
                data-slot="changelog-popup-highlight-text"
                className="text-foreground"
              >
                {item}
              </Text>
            </li>
          ))}
        </ul>

        <div
          data-slot="changelog-popup-footer"
          className="border-t border-border pt-3"
        >
          <Link
            href={latestEntry.url}
            isExternal
            onClick={dismiss}
            data-slot="changelog-popup-link"
            className="inline-flex items-center gap-1.5 text-foreground"
          >
            View changelog
            <Icon
              data-slot="changelog-popup-link-icon"
              icon={ArrowUpRight01Icon}
              size={16}
            />
          </Link>
        </div>
      </div>
    </div>
  );
}
