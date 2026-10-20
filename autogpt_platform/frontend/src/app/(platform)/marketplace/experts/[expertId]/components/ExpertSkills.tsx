import type { ExpertBundledSkill } from "@/app/api/__generated__/models/expertBundledSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { getCategoryAccent } from "../../../components/ExpertsSection/helpers";
import { ShelfTile } from "../../../components/Shelf/ShelfTile";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: ExpertBundledSkill[];
  /** The expert's area, whose glyph marks the skills as its work. */
  category?: string;
}

/** The same tile the marketplace's Skills shelf uses, so a skill here and a
 *  skill there are one object — and the description a pill had no room for
 *  gets its line. */
export function ExpertSkills({ skills, category }: Props) {
  const { accent, icon } = getCategoryAccent(category);

  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <ul className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {skills.map((skill) => (
          <ShelfTile
            key={skill.id}
            href={`/marketplace/skills/${skill.slug}`}
            mediaClassName={cn("h-10 w-10", accent.pill)}
            media={
              <Icon
                icon={icon ?? BookOpen01Icon}
                size={18}
                className={accent.icon}
                aria-hidden
              />
            }
            title={skill.title}
            subtitle={skill.description}
          />
        ))}
      </ul>
    </ExpertSection>
  );
}
