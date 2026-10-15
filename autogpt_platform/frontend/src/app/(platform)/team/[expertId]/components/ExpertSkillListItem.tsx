"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  BookOpen01Icon,
  Delete02Icon,
  LinkSquare02Icon,
} from "@hugeicons/core-free-icons";
import { ExpertSkillEntry } from "./useExpertSkills";

const VISIBLE_TRIGGERS = 3;

interface Props {
  entry: ExpertSkillEntry;
  accentClassName?: string;
  isSaving?: boolean;
  onRemove?: (name: string) => void;
}

/** A skill as a card: accent tile, name, two lines of description and the
 *  phrases that set it off. Actions sit top-right and fade in on hover
 *  (always shown where there is no hover). */
export function ExpertSkillListItem({
  entry,
  accentClassName,
  isSaving,
  onRemove,
}: Props) {
  const details = entry.skill ?? entry.library;
  const triggers = details?.triggers ?? [];
  const shown = triggers.slice(0, VISIBLE_TRIGGERS);
  const hiddenCount = triggers.length - shown.length;

  return (
    <div
      className="group relative flex h-full flex-col gap-3 rounded-2xl bg-white p-4 transition-shadow smooth-shadow-ring-sm hover:shadow-md"
      data-testid="expert-skill-row"
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl bg-violet-50 text-violet-700",
            accentClassName,
            "border-0",
          )}
        >
          <Icon icon={BookOpen01Icon} size={18} className="text-inherit" />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1 pr-16">
          <Text variant="body-medium" className="break-words !text-zinc-900">
            {entry.name}
          </Text>
          <Text variant="small" className="line-clamp-2 !text-zinc-500">
            {details?.description ?? "Skill details unavailable."}
          </Text>
        </div>
      </div>

      {shown.length > 0 ? (
        <div className="mt-auto flex flex-wrap gap-1.5">
          {shown.map((trigger) => (
            <span
              key={trigger}
              className="max-w-full truncate rounded-full border border-zinc-200 bg-zinc-50 px-2.5 py-0.5 text-xs text-zinc-600"
            >
              {trigger}
            </span>
          ))}
          {hiddenCount > 0 ? (
            <span className="rounded-full px-1.5 py-0.5 text-xs text-zinc-500">
              +{hiddenCount}
            </span>
          ) : null}
        </div>
      ) : null}

      <div className="absolute right-3 top-3 flex items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
        {entry.library ? (
          <Button
            as="NextLink"
            href="/library/skills"
            variant="ghost"
            size="icon-sm"
            aria-label="Open in library"
            leadingIcon={LinkSquare02Icon}
            className="text-zinc-500"
          />
        ) : null}
        {onRemove ? (
          <Button
            variant="ghost"
            size="icon-sm"
            leadingIcon={Delete02Icon}
            disabled={isSaving}
            onClick={() => onRemove(entry.name)}
            aria-label={`Remove ${entry.name}`}
            className="text-zinc-500 hover:text-red-600"
          />
        ) : null}
      </div>
    </div>
  );
}
