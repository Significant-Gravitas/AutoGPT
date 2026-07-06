"use client";

import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { PlusIcon } from "@phosphor-icons/react";
import { NEW_SKILL_PROMPT } from "../guidedPrompts";
import { EmptySkills } from "./components/EmptySkills/EmptySkills";
import { SkillListItem } from "./components/SkillListItem/SkillListItem";
import { UploadSkillButton } from "./components/UploadSkillButton/UploadSkillButton";
import { useSkillsPanel } from "./useSkillsPanel";

interface Props {
  onGuidedPrompt: (prompt: string) => void;
  withHeading?: boolean;
}

export function SkillsPanel({ onGuidedPrompt, withHeading = true }: Props) {
  const { skills, isLoading, error, newSkillName, handleSkillUploaded } =
    useSkillsPanel();

  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-col gap-2">
          {withHeading && <Text variant="h2">AutoPilot skills</Text>}
          <Text variant="body" className="!text-zinc-500">
            Reusable procedures your AutoPilot has distilled from past sessions.
            Review what it remembers, import a new skill, or delete one you no
            longer want it to reach for.
          </Text>
        </div>
        <div className="flex flex-shrink-0 items-center gap-2">
          <UploadSkillButton onUploaded={handleSkillUploaded} />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="primary"
                size="small"
                onClick={() => onGuidedPrompt(NEW_SKILL_PROMPT)}
                data-testid="skill-new-button"
              >
                <PlusIcon className="mr-1 h-4 w-4" />
                New skill
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              Teach AutoPilot a new skill in chat
            </TooltipContent>
          </Tooltip>
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
              <SkillListItem
                skill={skill}
                isNew={skill.name === newSkillName}
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
