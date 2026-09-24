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

/** A category worn as a label rather than a filter: same chip, no click. */
export function CategoryTag({ category, size = "small", className }: Props) {
  const color = getCategoryHex(category);
  const { accent, icon } = getCategoryAccent(category);

  return (
    <span className={cn(CHIP_SHAPE, CHIP_SIZE[size], "max-w-full", className)}>
      {color && (
        <span
          aria-hidden="true"
          className="size-2 shrink-0 rounded-full ring-1 ring-black/10"
          style={{ backgroundColor: color }}
        />
      )}
      {icon ? (
        <Icon
          icon={icon}
          size={TAG_ICON_SIZE[size]}
          style={color ? { color } : undefined}
          className={cn("shrink-0", accent.icon)}
          aria-hidden
        />
      ) : null}
      <span className="truncate">{formatCategoryLabel(category)}</span>
    </span>
  );
}
