import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { useId } from "react";
import { getReleaseLabel, Release } from "../helpers";
import {
  RELEASE_DETAILS_GAP_CLASS,
  RELEASE_DOT_COLUMN_CLASS,
  RELEASE_HEADLINE_GAP_CLASS,
  RELEASE_ROW_CLASS,
} from "./releaseRowLayout";

// Text past this many characters may hit the line clamp, so it also gets a
// tooltip with the whole of it.
const CLAMP_HINT_CHARS = 100;

interface Props {
  release: Release;
  isLatest: boolean;
  isLast: boolean;
}

export function ReleaseItem({ release, isLatest, isLast }: Props) {
  const detailsID = useId();
  const { headline, details } = release;

  return (
    <li className="relative">
      {isLast ? null : (
        <span
          aria-hidden
          className="pointer-events-none absolute -bottom-4 left-[1.15625rem] top-6 z-10 w-px bg-border-base"
        />
      )}
      <a
        href={release.url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={getReleaseLabel(release, isLatest)}
        aria-describedby={details ? detailsID : undefined}
        className={cn(
          RELEASE_ROW_CLASS,
          "group rounded-xl transition-colors duration-150 hover:bg-surface-hover/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info motion-reduce:transition-none",
        )}
      >
        <span aria-hidden className={RELEASE_DOT_COLUMN_CLASS}>
          <span
            className={cn(
              "size-2 rounded-full",
              isLatest
                ? "bg-brand ring-4 ring-accent-surface"
                : "bg-border-strong",
            )}
          />
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="flex items-center gap-2">
            <Text as="span" variant="small-medium-tight" tone="secondary">
              {release.dateRange}
            </Text>
            {isLatest ? (
              <span className="rounded-full bg-accent-surface px-2 py-px font-sans text-caption font-medium leading-4 text-accent-strong">
                Latest
              </span>
            ) : null}
            <Icon
              icon={ArrowUpRight01Icon}
              size={16}
              aria-hidden
              className="ml-auto shrink-0 text-foreground-tertiary transition-[color,transform] duration-150 group-hover:-translate-y-px group-hover:translate-x-px group-hover:text-foreground-strong motion-reduce:transition-none motion-reduce:group-hover:transform-none"
            />
          </span>
          <Text
            as="span"
            variant="body-medium"
            tone="primary"
            title={headline.length > CLAMP_HINT_CHARS ? headline : undefined}
            className={cn(
              RELEASE_HEADLINE_GAP_CLASS,
              "line-clamp-3 [overflow-wrap:anywhere]",
            )}
          >
            {headline}
          </Text>
          {details ? (
            <Text
              as="span"
              id={detailsID}
              variant="body-tight"
              tone="secondary"
              title={details.length > CLAMP_HINT_CHARS ? details : undefined}
              className={cn(
                RELEASE_DETAILS_GAP_CLASS,
                "line-clamp-3 [overflow-wrap:anywhere]",
              )}
            >
              {details}
            </Text>
          ) : null}
        </span>
      </a>
    </li>
  );
}
