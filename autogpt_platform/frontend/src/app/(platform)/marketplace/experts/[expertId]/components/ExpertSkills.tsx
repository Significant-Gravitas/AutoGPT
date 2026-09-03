import { ExpertSectionLabel } from "./ExpertSectionLabel";

interface Props {
  skills: string[];
}

export function ExpertSkills({ skills }: Props) {
  if (skills.length === 0) return null;

  return (
    <section>
      <ExpertSectionLabel>Skills</ExpertSectionLabel>
      <div className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <span
            key={skill}
            className="rounded-full bg-zinc-50 px-3 py-1.5 text-sm text-zinc-600 ring-1 ring-inset ring-zinc-200/80"
          >
            {skill}
          </span>
        ))}
      </div>
    </section>
  );
}
