"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import type { AsyncStatus } from "@/types/async-status";
import {
  ExpertAccent,
  getDayOneWorkflow,
  getExpertFirstName,
} from "../../helpers";
import { DayOneSection } from "./components/DayOneSection/DayOneSection";
import { ExpertProfileActions } from "./components/ExpertProfileActions/ExpertProfileActions";
import { ExpertProfileHeader } from "./components/ExpertProfileHeader/ExpertProfileHeader";
import { IncludedPlanSection } from "./components/IncludedPlanSection/IncludedPlanSection";
import { SkillsSection } from "./components/SkillsSection/SkillsSection";
import { WorkflowsSection } from "./components/WorkflowsSection/WorkflowsSection";

interface Props {
  expert: Expert;
  accent: ExpertAccent;
  isHired: boolean;
  isHiring: boolean;
  onHire: () => void;
  hiredExpertId: string | null;
  hiredLookup: AsyncStatus;
  onRetryHiredLookup: () => void;
}

export function ExpertProfileContent({
  expert,
  accent,
  isHired,
  isHiring,
  onHire,
  hiredExpertId,
  hiredLookup,
  onRetryHiredLookup,
}: Props) {
  const firstName = getExpertFirstName(expert.name);

  return (
    <div className="relative">
      <ExpertProfileHeader expert={expert} accent={accent} />

      <DayOneSection
        key={expert.id}
        firstName={firstName}
        firstWorkflow={getDayOneWorkflow(expert.workflows)}
        bio={expert.bio}
        accent={accent}
      />

      <SkillsSection skills={expert.skills} />
      <WorkflowsSection
        firstName={firstName}
        workflows={expert.workflows}
        accent={accent}
      />
      <IncludedPlanSection expertName={expert.name} accent={accent} />
      <ExpertProfileActions
        expertName={expert.name}
        isHired={isHired}
        isHiring={isHiring}
        onHire={onHire}
        hiredExpertId={hiredExpertId}
        hiredLookup={hiredLookup}
        onRetryHiredLookup={onRetryHiredLookup}
      />
    </div>
  );
}
