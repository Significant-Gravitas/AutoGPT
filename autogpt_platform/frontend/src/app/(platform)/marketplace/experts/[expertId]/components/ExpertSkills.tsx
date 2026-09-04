import { Badge } from "@/components/atoms/Badge/Badge";
import { ExpertSection } from "./ExpertSection";

interface Props {
  skills: string[];
}

export function ExpertSkills({ skills }: Props) {
  if (skills.length === 0) return null;

  return (
    <ExpertSection title="Skills">
      <div className="flex flex-wrap gap-1.5">
        {skills.map((skill) => (
          <Badge key={skill} variant="info" className="bg-white text-zinc-700">
            {skill}
          </Badge>
        ))}
      </div>
    </ExpertSection>
  );
}
