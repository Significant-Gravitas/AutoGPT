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
  const { accent, icon } = getCategoryAccent(category);

  return (
    <span className={cn(CHIP_SHAPE, CHIP_SIZE[size], "max-w-full", className)}>
      {icon ? (
        <Icon
          icon={icon}
          size={TAG_ICON_SIZE[size]}
          className={cn("shrink-0", accent.icon)}
          aria-hidden
        />
      ) : null}
      <span className="truncate">{formatCategoryLabel(category)}</span>
    </span>
  );
}
