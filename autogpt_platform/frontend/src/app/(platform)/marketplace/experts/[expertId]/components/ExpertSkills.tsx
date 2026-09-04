import { cn } from "@/lib/utils";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: string[];
  accent: ExpertAccent;
}

export function ExpertSkills({ skills, accent }: Props) {
  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <div className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <span
            key={skill}
            className={cn(
              "inline-flex items-center rounded-lg px-2.5 py-1 text-sm font-medium leading-5",
              accent.chip,
            )}
          >
            {skill}
          </span>
        ))}
      </div>
    </ExpertSection>
  );
}
