import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { Badge } from "@/components/atoms/Badge/Badge";
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
import {
  ArrowRight02Icon,
  Book04Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { CHIP_SHAPE, CHIP_SIZE } from "../../CategoryChip/CategoryChip";
import { formatCategoryLabel } from "../../SkillsSection/helpers";
import { getCategoryAccent, getExpertAccent } from "../helpers";
import { ExpertHireButton } from "./ExpertHireButton";

/** Three named, then a count for the rest — enough to place the expert
 *  without turning the card into a list. */
const NAMED_SKILLS = 3;

interface Props {
  expert: ExpertTemplate;
  isHired: boolean;
}

/** The whole card opens the expert's own page, through a link stretched
 *  behind its contents: a card that is itself a link cannot hold the hire
 *  button, since an anchor may not contain one. */
export function ExpertCard({ expert, isHired }: Props) {
  const accent = getExpertAccent(expert.role);
  const skills = expert.bundled_skills ?? [];
  const restSkills = skills.slice(NAMED_SKILLS);
  // The area an expert works in, in the same chip the filters use — the row
  // above the shelf and the card below it name the same thing.
  const area = expert.categories?.[0];
  const areaAccent = getCategoryAccent(area);

  return (
    <div className="group relative flex flex-col overflow-hidden rounded-2xl border border-zinc-200/80 bg-white text-left shadow-[0_1px_2px_rgba(16,24,40,0.04)] transition-all duration-200 ease-out hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]">
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 h-28 opacity-60 transition-opacity duration-200 group-hover:opacity-100",
          accent.wash,
        )}
      />
      <Link
        href={`/marketplace/experts/${expert.id}`}
        aria-label={`View ${expert.name}`}
        className="absolute inset-0 rounded-2xl outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
      />
      <div className="absolute right-5 top-5 z-10">
        {isHired ? (
          <Badge
            variant="success"
            className="rounded-full px-2.5 py-1 shadow-[0_1px_2px_rgba(16,24,40,0.05)]"
          >
            <Icon icon={CheckmarkCircle02Icon} size={14} />
            On your team
          </Badge>
        ) : (
          <ExpertHireButton expert={expert} />
        )}
      </div>
      {/* Inert, so a click anywhere lands on the link underneath; the pieces
          that answer to a pointer take their events back. */}
      <div className="pointer-events-none relative flex flex-1 flex-col gap-4 p-6">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          size={88}
        />

        <div>
          {/* The job title trails the name rather than taking the chip, which
              the expert's area has earned. */}
          <ExpertIdentityDetails
            name={expert.name}
            nameAlign="baseline"
            nameAccessory={
              expert.job_title ? (
                <span className="min-w-0 truncate text-sm text-zinc-500">
                  {expert.job_title}
                </span>
              ) : undefined
            }
          />
          {area ? (
            <span
              className={cn(CHIP_SHAPE, CHIP_SIZE.small, "mt-2 max-w-full")}
            >
              {areaAccent.icon ? (
                <Icon
                  icon={areaAccent.icon}
                  size={12}
                  className={cn("shrink-0", areaAccent.accent.icon)}
                  aria-hidden
                />
              ) : null}
              <span className="truncate">{formatCategoryLabel(area)}</span>
            </span>
          ) : null}
          <ExpertTagline tagline={expert.tagline} compact />
        </div>

        {skills.length > 0 ? (
          <div className="flex flex-wrap items-center gap-x-4">
            {/* The same chip the filters wear, so a skill on a card and a
                topic in the header read as one family. */}
            {skills.slice(0, NAMED_SKILLS).map((skill) => (
              <span
                key={skill.id}
                className={cn(
                  CHIP_SHAPE,
                  CHIP_SIZE.small,
                  "h-6 max-w-full border-transparent px-0",
                )}
              >
                <Icon
                  icon={Book04Icon}
                  size={12}
                  className="shrink-0 text-zinc-400"
                  aria-hidden
                />
                <span className="truncate">{skill.title}</span>
              </span>
            ))}
            {restSkills.length > 0 ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={cn(
                      CHIP_SHAPE,
                      CHIP_SIZE.small,
                      "pointer-events-auto h-6 cursor-default border-transparent px-0 text-zinc-500",
                    )}
                  >
                    +{restSkills.length} skills
                  </span>
                </TooltipTrigger>
                {/* Portalled: the card clips its own overflow. */}
                <TooltipPortal>
                  <TooltipContent side="top">
                    <ul className="space-y-0.5">
                      {restSkills.map((skill) => (
                        <li key={skill.id}>{skill.title}</li>
                      ))}
                    </ul>
                  </TooltipContent>
                </TooltipPortal>
              </Tooltip>
            ) : null}
          </div>
        ) : null}

        <div className="mt-auto flex items-center justify-end pt-2">
          <span className="flex items-center gap-1.5 text-base font-medium text-zinc-400 transition-colors duration-200 group-hover:text-zinc-900">
            View
            <Icon
              icon={ArrowRight02Icon}
              size={16}
              className="transition-transform duration-200 group-hover:translate-x-0.5"
            />
          </span>
        </div>
      </div>
    </div>
  );
}
