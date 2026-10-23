import type { CSSProperties } from "react";
import { getCategoryHex } from "@/components/molecules/ExpertAvatar/colors";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { getCategoryAccent } from "../ExpertsSection/helpers";
import { formatCategoryLabel } from "../SkillsSection/helpers";
import { CHIP_SHAPE, CHIP_SIZE, type ChipSize } from "./CategoryChip";

const TAG_ICON_SIZE: Record<ChipSize, number> = {
  small: 12,
  medium: 13,
  default: 15,
};

/** The category colour, pulled darker for the text and icon and washed out
 *  for the fill and edge, so the tag reads clearly on white. */
export function categoryTagStyle(color: string): CSSProperties {
  return {
    color: `color-mix(in srgb, ${color} 55%, rgb(var(--color-secondary-foreground)))`,
    backgroundColor: `color-mix(in srgb, ${color} 12%, rgb(var(--color-card)))`,
    // TODO(tokens): no fitting role for white #FFFFFF; #FFFFFF is card (a surface fill), popover (a surface fill), accent-foreground (a text colour)
    borderColor: `color-mix(in srgb, ${color} 35%, white)`,
  };
}

interface Props {
  category: string;
  size?: ChipSize;
  className?: string;
}

/** A category worn as a label rather than a filter: same chip, no click,
 *  tinted with the category's colour (see categoryTagStyle). */
export function CategoryTag({ category, size = "small", className }: Props) {
  const color = getCategoryHex(category);
  const { accent, icon } = getCategoryAccent(category);

  return (
    <span
      className={cn(
        CHIP_SHAPE,
        CHIP_SIZE[size],
        // A label, not a button: shorter than the clickable chip.
        size === "small" && "h-[22px]",
        "max-w-full",
        accent.icon,
        className,
      )}
      style={color ? categoryTagStyle(color) : undefined}
    >
      {icon ? (
        <Icon
          icon={icon}
          size={TAG_ICON_SIZE[size]}
          className="shrink-0"
          aria-hidden
        />
      ) : null}
      <span className="truncate">{formatCategoryLabel(category)}</span>
    </span>
  );
}
