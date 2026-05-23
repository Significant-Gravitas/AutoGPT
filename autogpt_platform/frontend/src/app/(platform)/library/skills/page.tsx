"use client";

import { useEffect } from "react";
import { notFound } from "next/navigation";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { EmptySkills } from "./components/EmptySkills/EmptySkills";
import { SkillListItem } from "./components/SkillListItem/SkillListItem";
import { useSkillsPage } from "./useSkillsPage";

export default function SkillsPage() {
  const isEnabled = useGetFlag(Flag.COPILOT_SKILLS_FOLLOWUPS);
  const { skills, isLoading, error } = useSkillsPage();

  if (!isEnabled) {
    notFound();
  }

  useEffect(() => {
    document.title = "Copilot skills – AutoGPT Platform";
  }, []);

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <header className="flex flex-col gap-2">
        <Text variant="h2">Copilot skills</Text>
        <Text variant="body" className="!text-zinc-500">
          Reusable procedures your copilot has distilled from past sessions.
          Review what it remembers, or delete a skill you no longer want it to
          reach for.
        </Text>
      </header>

      {error ? (
        <ErrorCard
          responseError={{
            message:
              error instanceof Error ? error.message : "Failed to load skills",
          }}
          context="copilot skills"
        />
      ) : isLoading ? (
        <div
          className="flex items-center justify-center py-16"
          data-testid="skills-loading"
        >
          <LoadingSpinner />
        </div>
      ) : skills.length === 0 ? (
        <EmptySkills />
      ) : (
        <ul
          className="flex flex-col gap-3"
          data-testid="skills-list"
          aria-label="Copilot skills"
        >
          {skills.map((skill) => (
            <li key={skill.name}>
              <SkillListItem skill={skill} />
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
