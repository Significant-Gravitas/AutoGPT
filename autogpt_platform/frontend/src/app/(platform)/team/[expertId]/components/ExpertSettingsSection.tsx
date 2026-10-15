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
      className="overflow-hidden rounded-2xl bg-white smooth-shadow-ring-sm"
    >
      <div className="flex items-start gap-3 border-b border-red-100 bg-red-50/60 px-5 py-4">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-red-100 text-red-600">
          <Icon icon={UserRemove01Icon} size={20} aria-hidden />
        </span>
        <div className="min-w-0">
          <Text variant="large-medium" className="!text-zinc-900">
            Fire {expert.name}
          </Text>
          <Text variant="body" className="mt-0.5 !text-zinc-600">
            {expert.name} leaves your team and stops working for you. Here is
            what that means for what you have set up.
          </Text>
        </div>
      </div>

      <ul className="divide-y divide-zinc-100 px-5">
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
              className="mt-1 text-sm font-medium text-zinc-700 underline underline-offset-2 hover:text-zinc-950"
            >
              Retry
            </button>
          ) : null}
          {pauseNames.length > 0 ? (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {pauseNames.map((item) => (
                <span
                  key={item.id}
                  className="max-w-full truncate rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs text-amber-800"
                >
                  {item.name}
                </span>
              ))}
              {hiddenPauseCount > 0 ? (
                <span className="px-1 py-0.5 text-xs text-zinc-500">
                  +{hiddenPauseCount} more
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

      <div className="flex justify-end border-t border-zinc-100 bg-zinc-50/60 px-5 py-3">
        <Button
          variant="destructive"
          size="small"
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
            ? "border-amber-200 bg-amber-50 text-amber-600"
            : "border-zinc-200 bg-zinc-50 text-zinc-500",
        )}
      >
        <Icon icon={icon} size={15} aria-hidden />
      </span>
      <div className="min-w-0 flex-1 pt-1">
        <div className="flex items-center justify-between gap-3">
          <Text variant="body" className="!text-zinc-800">
            {title}
          </Text>
          {tag ? (
            <span
              className={cn(
                "shrink-0 rounded-full px-2 py-0.5 text-xs font-medium",
                isWarn
                  ? "bg-amber-50 text-amber-700"
                  : "bg-emerald-50 text-emerald-700",
              )}
            >
              {tag}
            </span>
          ) : null}
        </div>
        {children}
      </div>
    </li>
  );
}
