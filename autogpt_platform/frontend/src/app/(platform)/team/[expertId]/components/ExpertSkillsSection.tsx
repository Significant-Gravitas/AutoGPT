"use client";

interface Props {
  expertName: string;
  skills: string[];
}

export function ExpertSkillsSection({ expertName, skills }: Props) {
  if (skills.length === 0) {
    return (
      <p className="text-sm text-zinc-500">
        No skills listed yet. {expertName} picks these up from the workflows you
        install.
      </p>
    );
  }

  return (
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
  );
}
