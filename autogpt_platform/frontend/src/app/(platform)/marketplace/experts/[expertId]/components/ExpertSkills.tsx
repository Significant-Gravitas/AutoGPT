import type { ExpertBundledSkill } from "@/app/api/__generated__/models/expertBundledSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { formatSkillTitle } from "../../../components/SkillsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: ExpertBundledSkill[];
  accent: ExpertAccent;
}

export function ExpertSkills({ skills, accent }: Props) {
  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <div className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <Link
            key={skill.id}
            href={`/marketplace/skills/${skill.slug}`}
            className={cn(
              "inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-sm font-medium leading-5 underline-offset-2 hover:underline",
              accent.chip,
            )}
          >
            {formatSkillTitle(skill.name)}
            <Icon icon={ArrowUpRight01Icon} size={14} aria-hidden />
          </Link>
        ))}
      </div>
    </ExpertSection>
  );
}
