import type { ExpertBundledSkill } from "@/app/api/__generated__/models/expertBundledSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { getCategoryAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertPill } from "./ExpertPill";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: ExpertBundledSkill[];
  /** The expert's area, whose glyph marks the skills as its work. */
  category?: string;
}

export function ExpertSkills({ skills, category }: Props) {
  const { accent, icon } = getCategoryAccent(category);

  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <ul className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <ExpertPill
            key={skill.id}
            icon={
              <Icon
                icon={icon ?? BookOpen01Icon}
                size={16}
                className={accent.icon}
              />
            }
            label={skill.title}
            href={`/marketplace/skills/${skill.slug}`}
          />
        ))}
      </ul>
    </ExpertSection>
  );
}
