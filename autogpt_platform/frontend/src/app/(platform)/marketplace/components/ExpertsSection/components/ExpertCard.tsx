import {
  expertPastel,
  getExpertTopicHex,
} from "@/components/molecules/ExpertAvatar/colors";
import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { ExpertTagline } from "@/components/molecules/ExpertIdentityDetails/components/ExpertTagline";
import { cn } from "@/lib/utils";
import { ArrowRight02Icon, Book04Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { CHIP_SHAPE, CHIP_SIZE } from "../../CategoryChip/CategoryChip";
import { CategoryTag } from "../../CategoryChip/CategoryTag";
import { getCategoryAccent } from "../helpers";
import { ExpertHireControl } from "./ExpertHireControl";

/** Three named, then a count for the rest — enough to place the expert
 *  without turning the card into a list. */
const NAMED_SKILLS = 3;

interface Props {
  expert: ExpertTemplate;
  isHired: boolean;
  category?: string | null;
}

/** The whole card opens the expert's own page, through a link stretched
 *  behind its contents: a card that is itself a link cannot hold the hire
 *  button, since an anchor may not contain one. */
export function ExpertCard({ expert, isHired, category }: Props) {
  const selectedCategory =
    category && expert.categories?.includes(category) ? category : undefined;
  // The expert's own visual family, whatever filter the shelf is under: the
  // same Maria under Marketing and under Content.
  const topicColor = getExpertTopicHex({
    avatarUrl: expert.avatar_url,
    categories: expert.categories,
    role: expert.role,
  });
  const skills = expert.bundled_skills ?? [];
  const named = skills.slice(0, NAMED_SKILLS);
  const restSkills = skills.slice(NAMED_SKILLS);
  // The area an expert works in, in the same chip the filters use — the row
  // above the shelf and the card below it name the same thing.
  const area = selectedCategory ?? expert.categories?.[0];
  const areaAccent = getCategoryAccent(area);

  return (
    <div className="group relative flex flex-col overflow-hidden rounded-2xl border border-zinc-200/80 bg-white text-left shadow-[0_1px_2px_rgba(16,24,40,0.04)] transition-all duration-200 ease-out hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]">
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[4.5rem]"
        style={{ backgroundColor: expertPastel(topicColor) }}
      />
      <Link
        href={`/marketplace/experts/${expert.id}`}
        className="relative flex flex-1 flex-col gap-4 rounded-2xl p-6 outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
      >
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          size={88}
          backgroundColor={topicColor}
          className="rounded-full ring-4 ring-white"
        />

        <div>
          <ExpertIdentityDetails
            name={expert.name}
            role={expert.role}
            jobTitle={expert.job_title}
            nameAlign="baseline"
          />
          {area ? <CategoryTag category={area} className="mt-2" /> : null}
          <ExpertTagline tagline={expert.tagline} compact />
        </div>

        {skills.length > 0 ? (
          <div>
            <div className="mb-1 text-sm text-zinc-500">Skills:</div>
            <div className="flex flex-col items-start">
              {named.map((skill) => (
                <span
                  key={skill.id}
                  className={cn(
                    CHIP_SHAPE,
                    CHIP_SIZE.small,
                    "h-6 max-w-full border-transparent px-0",
                  )}
                >
                  <Icon
                    icon={areaAccent.icon ?? Book04Icon}
                    size={12}
                    className="shrink-0"
                    style={{ color: topicColor }}
                    aria-hidden
                  />
                  <span className="truncate">{skill.title}</span>
                </span>
              ))}
              {restSkills.length > 0 ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span
                      tabIndex={0}
                      className={cn(
                        CHIP_SHAPE,
                        CHIP_SIZE.small,
                        "h-6 cursor-default border-transparent px-0 text-zinc-500 outline-none focus-visible:ring-2 focus-visible:ring-violet-600",
                      )}
                    >
                      +{restSkills.length} skills
                    </span>
                  </TooltipTrigger>
                  <TooltipPortal>
                    <TooltipContent side="top">
                      <ul className="space-y-0.5">
                        {restSkills.map((rest) => (
                          <li key={rest.id}>{rest.title}</li>
                        ))}
                      </ul>
                    </TooltipContent>
                  </TooltipPortal>
                </Tooltip>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="mt-auto flex items-center justify-end pt-2">
          <span className="flex items-center gap-1.5 text-base font-medium text-zinc-400 transition-colors duration-200 group-hover:text-zinc-900">
            View
            <Icon
              icon={ArrowRight02Icon}
              size={16}
              aria-hidden
              className="transition-transform duration-200 group-hover:translate-x-0.5"
            />
          </span>
        </div>
      </Link>
      <div className="absolute right-5 top-5 z-10">
        <ExpertHireControl expert={expert} isHired={isHired} />
      </div>
    </div>
  );
}
