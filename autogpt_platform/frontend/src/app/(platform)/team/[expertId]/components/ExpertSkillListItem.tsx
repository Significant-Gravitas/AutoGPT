"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { BookOpen01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { ExpertSkillEntry } from "./useExpertSkills";

interface Props {
  entry: ExpertSkillEntry;
  accentClassName?: string;
  isSaving: boolean;
  onRemove: (name: string) => void;
}

export function ExpertSkillListItem({
  entry,
  accentClassName,
  isSaving,
  onRemove,
}: Props) {
  const triggers = entry.library?.triggers ?? [];
  return (
    <div
      className="flex w-full flex-col gap-3 rounded-large border border-zinc-200 bg-white p-4 sm:flex-row sm:items-start sm:justify-between"
      data-testid="expert-skill-row"
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <div
          className={cn(
            "flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-large bg-violet-50 text-violet-700",
            accentClassName,
            "border-0",
          )}
        >
          <Icon icon={BookOpen01Icon} size={18} className="text-inherit" />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="body-medium" className="break-words">
            {entry.name}
          </Text>
          <Text variant="small" className="!text-zinc-500">
            {entry.library?.description ??
              "Marketplace skill. Not in your library yet."}
          </Text>
          {triggers.length > 0 ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {triggers.map((trigger) => (
                <span
                  key={trigger}
                  className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600"
                >
                  {trigger}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      <div className="flex flex-shrink-0 items-center gap-2">
        {entry.library ? (
          <Button
            as="NextLink"
            href="/library/skills"
            variant="ghost"
            size="small"
          >
            Open in library
          </Button>
        ) : null}
        <Button
          variant="secondary"
          size="small"
          disabled={isSaving}
          onClick={() => onRemove(entry.name)}
          aria-label={`Remove ${entry.name}`}
        >
          <Icon icon={Delete02Icon} className="mr-1 h-4 w-4" />
          Remove
        </Button>
      </div>
    </div>
  );
}
