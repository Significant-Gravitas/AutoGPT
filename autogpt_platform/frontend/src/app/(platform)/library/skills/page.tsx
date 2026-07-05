"use client";

import { useEffect } from "react";
import Link from "next/link";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { EmptySkills } from "./components/EmptySkills/EmptySkills";
import { SkillListItem } from "./components/SkillListItem/SkillListItem";
import { UploadSkillButton } from "./components/UploadSkillButton/UploadSkillButton";
import { useSkillsPage } from "./useSkillsPage";

export default function SkillsPage() {
  const { skills, isLoading, error } = useSkillsPage();

  useEffect(() => {
    document.title = "AutoPilot skills – AutoGPT Platform";
  }, []);

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <Link
        href="/library"
        className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
        data-testid="skills-back-to-library"
      >
        <ArrowLeftIcon size={14} weight="bold" />
        Back to Library
      </Link>
      <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex flex-col gap-2">
          <Text variant="h2">AutoPilot skills</Text>
          <Text variant="body" className="!text-zinc-500">
            Reusable procedures your AutoPilot has distilled from past sessions.
            Review what it remembers, upload your own, download one to share, or
            delete a skill you no longer want it to reach for.
          </Text>
        </div>
        <div className="flex-shrink-0">
          <UploadSkillButton />
        </div>
      </header>

      {error ? (
        <ErrorCard
          responseError={{
            message:
              error instanceof Error ? error.message : "Failed to load skills",
          }}
          context="AutoPilot skills"
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
          aria-label="AutoPilot skills"
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
