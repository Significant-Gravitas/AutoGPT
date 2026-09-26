"use client";

import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import type { BeatKey } from "../../../flowItems";
import { VOICE_SAMPLES } from "../../../helpers";
import type { useRaisePage } from "../../../useRaisePage";
import { AboutStep } from "../../AboutStep/AboutStep";
import { AvatarStep } from "../../AvatarStep/AvatarStep";
import { BudgetStep } from "../../BudgetStep/BudgetStep";
import { CategoryStep } from "../../CategoryStep/CategoryStep";
import {
  interactiveCardClassFor,
  selectedCardClassFor,
  textClassFor,
} from "../../ColorStep/helpers";
import { JobTitleStep } from "../../JobTitleStep/JobTitleStep";
import { MarketplaceStep } from "../../MarketplaceStep/MarketplaceStep";
import { NameStep } from "../../NameStep/NameStep";
import {
  jobTitleSuggestionsFor,
  nameSuggestionsFor,
} from "../../RoleStep/helpers";
import { RoleStep } from "../../RoleStep/RoleStep";
import { SkillsStep } from "../../SkillsStep/SkillsStep";

interface Props {
  beat: BeatKey;
  flow: ReturnType<typeof useRaisePage>;
}

export function BeatControl({ beat, flow }: Props) {
  switch (beat) {
    case "role":
      return (
        <RoleStep
          selectedRole={flow.role}
          color={flow.color}
          onPick={flow.pickRole}
        />
      );
    case "jobTitle":
      return (
        <JobTitleStep
          selectedTitle={flow.jobTitle}
          suggestions={jobTitleSuggestionsFor(flow.role)}
          color={flow.color}
          onSubmit={flow.submitJobTitle}
          onSkip={flow.skipJobTitle}
        />
      );
    case "name":
      return (
        <NameStep
          selectedName={flow.name || null}
          suggestions={nameSuggestionsFor(flow.role)}
          color={flow.color}
          onSubmit={flow.submitName}
        />
      );
    case "category":
      return (
        <CategoryStep
          selectedCategory={flow.category}
          suggested={flow.suggestedCategory}
          color={flow.color}
          onPick={flow.pickCategory}
        />
      );
    case "avatar":
      return (
        <AvatarStep
          name={flow.name}
          category={flow.category ?? flow.suggestedCategory}
          color={flow.color}
          avatarUrl={flow.avatarUrl || null}
          onPick={flow.pickAvatar}
        />
      );
    case "about":
      return (
        <AboutStep
          submittedAbout={flow.about}
          name={flow.name}
          color={flow.color}
          onSubmit={flow.submitAbout}
          onSkip={flow.skipAbout}
        />
      );
    case "voice":
      return (
        <VoicePicker
          name={flow.name}
          samples={VOICE_SAMPLES}
          hideHeader
          labelClassName={textClassFor(flow.color)}
          cardColors={{
            selected: selectedCardClassFor(flow.color),
            interactive: interactiveCardClassFor(flow.color),
          }}
          onPick={flow.pickVoice}
          onSkip={flow.skipVoice}
        />
      );
    case "budget":
      return (
        <BudgetStep
          color={flow.color}
          submittedBudget={flow.budget}
          onSubmit={flow.submitBudget}
          onSkip={flow.skipBudget}
        />
      );
    case "marketplace":
      return (
        <MarketplaceStep
          name={flow.name}
          color={flow.color}
          submitted={flow.marketplace}
          isFinal={flow.isMarketplaceFinal}
          isSubmitting={flow.isSubmitting}
          onSubmit={flow.submitMarketplace}
          onSkip={flow.skipMarketplace}
        />
      );
    case "skills":
      return (
        <SkillsStep
          name={flow.name}
          color={flow.color}
          submitted={flow.skills}
          existingCount={flow.marketplace?.length ?? 0}
          isSubmitting={flow.isSubmitting}
          onSubmit={flow.submitSkills}
          onSkip={flow.skipSkills}
        />
      );
    default:
      // A new beat without a control fails to compile here rather than
      // silently rendering nothing.
      return assertUnreachable(beat);
  }
}

function assertUnreachable(beat: never): never {
  throw new Error(`No control registered for beat: ${String(beat)}`);
}
