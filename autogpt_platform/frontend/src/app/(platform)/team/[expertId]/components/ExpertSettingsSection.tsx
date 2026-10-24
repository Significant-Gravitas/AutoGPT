"use client";

import type { IconSvgElement } from "@hugeicons/react";
import {
  BubbleChatIcon,
  Calendar03Icon,
  Folder01Icon,
  Store01Icon,
  UserRemove01Icon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useExpertSettingsSection } from "./useExpertSettingsSection";

const VISIBLE_PAUSE_NAMES = 4;

interface Props {
  expert: Expert;
  onFire: () => void;
}

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** Firing, spelled out before anyone reaches for the button: what pauses,
 *  what stays, and that the expert can come back. */
export function ExpertSettingsSection({ expert, onFire }: Props) {
  const {
    pauseItems,
    isPreviewLoading,
    isPreviewError,
    retryPreview,
    fileCount,
    hasMoreFiles,
  } = useExpertSettingsSection(expert.id);
  const workflowCount = expert.workflows.length;
  const pauseNames = pauseItems.slice(0, VISIBLE_PAUSE_NAMES);
  const hiddenPauseCount = pauseItems.length - pauseNames.length;

  return (
    <section
      aria-label={`Fire ${expert.name}`}
      className="overflow-hidden rounded-2xl bg-surface smooth-shadow-ring-sm"
    >
      <div
        // TODO(tokens): no exact role: border-red-100 #FDECEC
        className="flex items-start gap-3 border-b border-red-100 bg-error-surface/60 px-5 py-4"
      >
        <span
          // TODO(tokens): no exact role: bg-red-100 #FDECEC; no fitting role for text-red-600 #D93636; #D93636 is error-hover (a hover fill)
          className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-red-100 text-red-600"
        >
          <Icon icon={UserRemove01Icon} size={20} aria-hidden />
        </span>
        <div className="min-w-0">
          <Text variant="large-medium" className="!text-foreground-strong">
            Fire {expert.name}
          </Text>
          <Text variant="body-md" className="mt-0.5 !text-foreground-secondary">
            {expert.name} leaves your team and stops working for you. Here is
            what that means for what you have set up.
          </Text>
        </div>
      </div>

      <ul className="divide-y divide-separator px-5">
        <Consequence
          icon={Calendar03Icon}
          tone={pauseItems.length > 0 || isPreviewError ? "warn" : "neutral"}
          tag={pauseItems.length > 0 ? "Pauses" : undefined}
          title={
            isPreviewLoading
              ? "Checking schedules and triggers…"
              : isPreviewError
                ? "Couldn't check which schedules and triggers would pause"
                : pauseItems.length > 0
                  ? `${plural(pauseItems.length, "automation")} will pause`
                  : "No schedules or triggers to pause"
          }
        >
          {isPreviewError ? (
            <button
              type="button"
              onClick={() => void retryPreview()}
              className="mt-1 inline-flex text-foreground-body underline underline-offset-2 hover:text-foreground"
            >
              <Text variant="body-md-medium" as="span" tone="inherit">
                Retry
              </Text>
            </button>
          ) : null}
          {pauseNames.length > 0 ? (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {pauseNames.map((item) => (
                <span
                  key={item.id}
                  className="flex min-w-0 max-w-full rounded-full border border-warning-border bg-warning-surface px-2.5 py-0.5"
                >
                  <Text
                    variant="small-tight"
                    as="span"
                    tone="warning-strong"
                    className="truncate"
                  >
                    {item.name}
                  </Text>
                </span>
              ))}
              {hiddenPauseCount > 0 ? (
                <span className="flex px-1 py-0.5">
                  <Text variant="small-tight" as="span" tone="muted">
                    +{hiddenPauseCount} more
                  </Text>
                </span>
              ) : null}
            </div>
          ) : null}
        </Consequence>
        <Consequence
          icon={WorkflowSquare01Icon}
          tag="Kept"
          title={
            workflowCount > 0
              ? `${plural(workflowCount, "workflow")} stay in your library`
              : "Installed workflows stay in your library"
          }
        />
        <Consequence
          icon={Folder01Icon}
          tag="Kept"
          title={
            fileCount > 0
              ? `${hasMoreFiles ? `${fileCount}+ files` : plural(fileCount, "file")} from its chats ${fileCount === 1 && !hasMoreFiles ? "stays" : "stay"} in your workspace`
              : "Files from its chats stay in your workspace"
          }
        />
        <Consequence
          icon={BubbleChatIcon}
          tag="Kept"
          title="Chat history stays, read-only"
        />
        <Consequence
          icon={Store01Icon}
          title={`You can hire ${expert.name} again anytime from the marketplace`}
        />
      </ul>

      <div className="flex justify-end border-t border-separator bg-surface-muted/60 px-5 py-3">
        <Button
          variant="destructive"
          size="small"
          textSize="body-md"
          onClick={onFire}
          data-testid="expert-fire-button"
        >
          Fire {expert.name}
        </Button>
      </div>
    </section>
  );
}

interface ConsequenceProps {
  icon: IconSvgElement;
  title: string;
  tag?: string;
  tone?: "neutral" | "warn";
  children?: ReactNode;
}

function Consequence({
  icon,
  title,
  tag,
  tone = "neutral",
  children,
}: ConsequenceProps) {
  const isWarn = tone === "warn";
  return (
    <li className="flex items-start gap-3 py-3">
      <span
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-lg border",
          isWarn
            ? // TODO(tokens): no exact role: text-amber-600 #D97706
              "border-warning-border bg-warning-surface text-amber-600"
            : "border-border-base bg-surface-muted text-foreground-tertiary",
        )}
      >
        <Icon icon={icon} size={15} aria-hidden />
      </span>
      <div className="min-w-0 flex-1 pt-1">
        <div className="flex items-center justify-between gap-3">
          <Text variant="body-md" className="!text-foreground-emphasis">
            {title}
          </Text>
          {tag ? (
            <span
              className={cn(
                "flex shrink-0 rounded-full px-2 py-0.5",
                isWarn
                  ? "bg-warning-surface"
                  : // TODO(tokens): no exact role: bg-emerald-50 #ECFDF5
                    "bg-emerald-50",
              )}
            >
              <Text
                variant="small-medium-tight"
                as="span"
                tone={isWarn ? "warning" : "success-strong"}
              >
                {tag}
              </Text>
            </span>
          ) : null}
        </div>
        {children}
      </div>
    </li>
  );
}
