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

interface Props {
  category: string;
  size?: ChipSize;
  className?: string;
}

/** A category worn as a label rather than a filter: same chip, no click.
 *  The glyph carries the category's colour and the label takes it from
 *  `currentColor`, so the tag needs no separate swatch. */
export function CategoryTag({ category, size = "small", className }: Props) {
  const color = getCategoryHex(category);
  const { accent, icon } = getCategoryAccent(category);

  return (
    <span
      className={cn(
        CHIP_SHAPE,
        CHIP_SIZE[size],
        "max-w-full",
        accent.icon,
        className,
      )}
      style={color ? { color } : undefined}
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
