import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Book04Icon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { getCategoryAccent } from "../../ExpertsSection/helpers";
import { ShelfTile } from "../../Shelf/ShelfTile";

interface Props {
  skill: MarketplaceSkill;
  isInstalled: boolean;
  onSee: () => void;
}

export function SkillTile({ skill, isInstalled, onSee }: Props) {
  const { accent } = getCategoryAccent(skill.categories[0]);

  return (
    <ShelfTile
      testId="skill-tile"
      onClick={onSee}
      mediaClassName={cn("h-10 w-10", accent.pill)}
      media={
        <Icon icon={Book04Icon} size={18} className={accent.icon} aria-hidden />
      }
      title={skill.title}
      subtitle={skill.description}
      trailing={
        isInstalled ? (
          <span className="flex shrink-0 items-center gap-1 pr-1 text-[13px] font-medium text-emerald-600">
            <Icon icon={CheckmarkCircle02Icon} size={14} aria-hidden />
            Added
          </span>
        ) : null
      }
    />
  );
}
