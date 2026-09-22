"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useStoreCategories } from "@/hooks/useStoreCategories";
import { cn } from "@/lib/utils";
import { getCategoryAccent } from "../ExpertsSection/helpers";

type Size = "small" | "default";

/** The zinc fallback `getCategoryAccent` hands back for "All", research and
 *  development. Its glossy chip is too faint to read as selected next to a
 *  white row, so those chips take a plain fill instead. */
const NEUTRAL_ACCENT = getCategoryAccent(undefined).accent;
const NEUTRAL_SELECTED = "border-transparent bg-zinc-100 text-zinc-900";

interface Props {
  selected: string | null;
  onSelect: (topic: string | null) => void;
  /** "small" tucks the row into a section header; "default" is a page's own
   *  filter and matches the height of the marketplace category pills. */
  size?: Size;
}

export function SkillTopicChips({ selected, onSelect, size = "small" }: Props) {
  const { categories } = useStoreCategories();

  if (categories.length === 0) return null;

  return (
    <div
      role="group"
      aria-label="Filter skills by topic"
      className={cn(
        "flex flex-wrap",
        size === "small" ? "gap-1 sm:justify-end" : "gap-2",
      )}
    >
      <TopicChip
        label="All"
        size={size}
        isSelected={selected === null}
        onClick={() => onSelect(null)}
      />
      {categories.map((category) => (
        <TopicChip
          key={category.value}
          topic={category.value}
          label={category.label}
          title={category.description}
          size={size}
          isSelected={selected === category.value}
          onClick={() =>
            onSelect(selected === category.value ? null : category.value)
          }
        />
      ))}
    </div>
  );
}

interface ChipProps {
  topic?: string;
  label: string;
  title?: string;
  size: Size;
  isSelected: boolean;
  onClick: () => void;
}

function TopicChip({
  topic,
  label,
  title,
  size,
  isSelected,
  onClick,
}: ChipProps) {
  const { accent, icon } = getCategoryAccent(topic);

  return (
    // Outline, so a row of filters never reads as a row of actions. The
    // selected one takes the accent's glossy chip, whose gradient sits over
    // the variant's hover wash rather than being replaced by it.
    <Button
      variant="outline"
      size={size === "small" ? "xs" : "small"}
      title={title}
      aria-pressed={isSelected}
      onClick={onClick}
      unmask={false}
      leftIcon={
        icon ? (
          <Icon
            icon={icon}
            size={size === "small" ? 12 : 15}
            className={isSelected ? undefined : accent.icon}
            aria-hidden
          />
        ) : undefined
      }
      className={cn(
        "min-w-0 rounded-full",
        size === "small" ? "gap-1 px-2" : "gap-1.5 px-3.5",
        isSelected &&
          (accent === NEUTRAL_ACCENT
            ? NEUTRAL_SELECTED
            : cn(accent.chip, "border-transparent")),
      )}
    >
      {label}
    </Button>
  );
}
