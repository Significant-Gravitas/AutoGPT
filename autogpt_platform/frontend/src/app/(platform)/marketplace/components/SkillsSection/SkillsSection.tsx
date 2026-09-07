"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Icon } from "@/components/atoms/Icon/Icon";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { SectionHeader } from "../SectionHeader";
import { SkillCard } from "./components/SkillCard";
import { useSkillsSection } from "./useSkillsSection";

export function SkillsSection() {
  const { skills, isLoading, isError } = useSkillsSection();

  if (isError || (!isLoading && skills.length === 0)) return null;

  return (
    <section id="skills" className="mb-20 scroll-mt-24">
      <SectionHeader
        titleIcon={<Icon icon={BookOpen01Icon} size={30} />}
        title="Skills to teach"
        subtitle="Instructions and examples your AutoPilot follows — a brand voice guide, an outreach playbook. Not something it runs; something it knows."
      />
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
        {isLoading
          ? [0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-48 w-full rounded-2xl" />
            ))
          : skills.map((skill) => <SkillCard key={skill.slug} skill={skill} />)}
      </div>
    </section>
  );
}
