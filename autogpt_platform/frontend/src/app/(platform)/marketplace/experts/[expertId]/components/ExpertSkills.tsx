import type { ExpertBundledSkill } from "@/app/api/__generated__/models/expertBundledSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: string[];
  bundledSkills: ExpertBundledSkill[];
  accent: ExpertAccent;
}

const CHIP_CLASS =
  "inline-flex items-center rounded-lg px-2.5 py-1 text-sm font-medium leading-5";

export function ExpertSkills({ skills, bundledSkills, accent }: Props) {
  if (skills.length === 0) return null;
  const bundled = new Map(bundledSkills.map((skill) => [skill.name, skill]));

  return (
    <ExpertSection title="Skills">
      <div className="flex flex-wrap gap-2">
        {skills.map((name) => {
          const skill = bundled.get(name);
          if (!skill) {
            return (
              <span key={name} className={cn(CHIP_CLASS, accent.chip)}>
                {name}
              </span>
            );
          }
          return (
            <Link
              key={name}
              href={`/marketplace/skills/${skill.slug}`}
              className={cn(
                CHIP_CLASS,
                accent.chip,
                "gap-1 underline-offset-2 hover:underline",
              )}
            >
              {skill.title}
              <Icon icon={ArrowUpRight01Icon} size={14} aria-hidden />
            </Link>
          );
        })}
      </div>
    </ExpertSection>
  );
}
