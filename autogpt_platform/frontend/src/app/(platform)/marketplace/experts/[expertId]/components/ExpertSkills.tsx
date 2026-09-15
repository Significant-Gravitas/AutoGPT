import type { ExpertBundledSkill } from "@/app/api/__generated__/models/expertBundledSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { formatSkillTitle } from "../../../components/SkillsSection/helpers";
import { ExpertPill } from "./ExpertPill";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: ExpertBundledSkill[];
  accent: ExpertAccent;
}

export function ExpertSkills({ skills, accent }: Props) {
  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <ul className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <ExpertPill
            key={skill.id}
            icon={
              <Icon icon={BookOpen01Icon} size={16} className={accent.icon} />
            }
            label={formatSkillTitle(skill.name)}
            href={`/marketplace/skills/${skill.slug}`}
          />
        ))}
      </ul>
    </ExpertSection>
  );
}
