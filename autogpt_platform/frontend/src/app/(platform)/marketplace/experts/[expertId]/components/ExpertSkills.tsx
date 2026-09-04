import { Badge } from "@/components/atoms/Badge/Badge";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: string[];
}

export function ExpertSkills({ skills }: Props) {
  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <div className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <Badge
            key={skill}
            variant="info"
            className="rounded-lg bg-white px-2.5 py-1 text-sm leading-5 text-zinc-700"
          >
            {skill}
          </Badge>
        ))}
      </div>
    </ExpertSection>
  );
}
