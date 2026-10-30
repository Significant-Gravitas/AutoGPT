"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import type { ReactNode } from "react";
import { EditBudgetDialog } from "../EditBudgetDialog/EditBudgetDialog";
import { ExpertAvatarButton } from "../ExpertAvatarButton/ExpertAvatarButton";
import { budgetLabel, countLabel, voiceLabel, VOICE_SAMPLES } from "./helpers";
import { useExpertSetupSection } from "./useExpertSetupSection";

interface Props {
  expert: Expert;
  onOpenTab: (tab: string) => void;
  onInstallWorkflow: () => void;
}

/** Everything the create dialog no longer asks: each starts on a sensible
 *  default and is one click from changed. */
export function ExpertSetupSection({
  expert,
  onOpenTab,
  onInstallWorkflow,
}: Props) {
  const setup = useExpertSetupSection(expert);

  return (
    <section
      aria-label={`${expert.name} setup`}
      className="overflow-hidden rounded-2xl bg-surface smooth-shadow-ring-sm"
    >
      <div className="border-b border-separator px-5 py-4">
        <Text variant="large-medium" tone="primary">
          Setup
        </Text>
        <Text variant="body-md" tone="secondary" className="mt-0.5">
          How {expert.name} looks, sounds and spends, and what they can run.
        </Text>
      </div>
      <ul className="divide-y divide-separator px-5">
        <SetupRow
          title="Weekly budget"
          value={budgetLabel(expert)}
          action={
            <Button size="small" variant="secondary" onClick={setup.openBudget}>
              Edit
            </Button>
          }
        />
        <SetupRow
          title="Voice"
          value={voiceLabel(expert)}
          action={
            <Button
              size="small"
              variant="secondary"
              onClick={() => setup.setIsVoiceOpen(true)}
            >
              Change
            </Button>
          }
        />
        <SetupRow
          title="Photo"
          value="Click the picture to change it"
          action={<ExpertAvatarButton expert={expert} isCompact />}
        />
        <SetupRow
          title="Workflows"
          value={countLabel(expert.workflows.length, "workflow")}
          action={
            <div className="flex gap-2">
              <Button
                size="small"
                variant="ghost"
                onClick={() => onOpenTab("workflows")}
              >
                View
              </Button>
              <Button
                size="small"
                variant="secondary"
                onClick={onInstallWorkflow}
              >
                Add workflow
              </Button>
            </div>
          }
        />
        <SetupRow
          title="Skills"
          value={countLabel(expert.skills.length, "skill")}
          action={
            <Button
              size="small"
              variant="secondary"
              onClick={() => onOpenTab("skills")}
            >
              Manage
            </Button>
          }
        />
      </ul>

      <EditBudgetDialog
        expert={expert}
        open={setup.isBudgetOpen}
        onClose={setup.closeBudget}
      />
      <Dialog
        title={`How should ${expert.name} sound?`}
        controlled={{ isOpen: setup.isVoiceOpen, set: setup.setIsVoiceOpen }}
      >
        <Dialog.Content>
          {setup.isVoiceOpen ? (
            <VoicePicker
              name={expert.name}
              samples={VOICE_SAMPLES}
              onPick={setup.pickVoice}
              onSkip={() => setup.setIsVoiceOpen(false)}
              isSubmitting={setup.isSavingVoice}
              hideHeader
            />
          ) : null}
        </Dialog.Content>
      </Dialog>
    </section>
  );
}

interface RowProps {
  title: string;
  value: string;
  action: ReactNode;
}

function SetupRow({ title, value, action }: RowProps) {
  return (
    <li className="flex min-h-16 items-center justify-between gap-4 py-3">
      <div className="min-w-0 tabular-nums">
        <Text variant="body-medium" tone="primary">
          {title}
        </Text>
        <Text variant="small" tone="muted" className="truncate">
          {value}
        </Text>
      </div>
      <div className="shrink-0">{action}</div>
    </li>
  );
}
