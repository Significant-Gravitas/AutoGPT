import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  BookOpen01Icon,
  CheckmarkCircle02Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { getCategoryAccent } from "../../ExpertsSection/helpers";

interface Props {
  skill: MarketplaceSkill;
  isInstalled: boolean;
  onSee: () => void;
}

export function SkillRow({ skill, isInstalled, onSee }: Props) {
  const { accent } = getCategoryAccent(skill.categories[0]);

  return (
    <li
      data-testid="skill-row"
      className="flex items-center gap-3 border-b border-zinc-200/70 py-3"
    >
      <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-inset ring-zinc-200/70">
        <Icon
          icon={BookOpen01Icon}
          size={18}
          className={accent.icon}
          aria-hidden
        />
      </span>
      <button
        type="button"
        onClick={onSee}
        className="min-w-0 flex-1 rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-violet-600 focus-visible:ring-offset-2"
      >
        <span className="block truncate text-sm font-medium text-zinc-900">
          {skill.title}
        </span>
        <span className="block truncate text-[13px] text-zinc-500">
          {skill.description}
        </span>
      </button>
      {isInstalled ? (
        <span className="flex shrink-0 items-center gap-1 text-[13px] font-medium text-emerald-600">
          <Icon icon={CheckmarkCircle02Icon} size={14} aria-hidden />
          Added
        </span>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={`More options for ${skill.title}`}
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 outline-none transition-colors hover:bg-zinc-100 hover:text-zinc-900 focus-visible:ring-2 focus-visible:ring-violet-600"
          >
            <Icon icon={MoreHorizontalIcon} size={18} aria-hidden />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={onSee}>See skill</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  );
}
