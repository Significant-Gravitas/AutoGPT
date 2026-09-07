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
  isSaving?: boolean;
  onRemove?: (name: string) => void;
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
      className="flex w-full flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-3 sm:flex-row sm:items-start sm:justify-between"
      data-testid="expert-skill-row"
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <div
          className={cn(
            "flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md bg-violet-50 text-violet-700",
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
          <Text variant="small" tone="muted">
            {entry.library?.description ??
              "Marketplace skill. Not in your library yet."}
          </Text>
          {triggers.length > 0 ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {triggers.map((trigger) => (
                <Text
                  key={trigger}
                  variant="small"
                  as="span"
                  tone="secondary"
                  className="rounded-md bg-zinc-100 px-2 py-0.5"
                >
                  {trigger}
                </Text>
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
            size="xs"
          >
            Open in library
          </Button>
        ) : null}
        {onRemove ? (
          <Button
            variant="secondary"
            size="xs"
            leadingIcon={Delete02Icon}
            disabled={isSaving}
            onClick={() => onRemove(entry.name)}
            aria-label={`Remove ${entry.name}`}
          >
            Remove
          </Button>
        ) : null}
      </div>
    </div>
  );
}
