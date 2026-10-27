"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { DEFAULT_AUTOPILOT_MODE } from "@/app/(platform)/copilot/autopilotModeStore";
import { AUTOPILOT_MODE_OPTIONS } from "@/app/(platform)/copilot/components/ChatInput/components/AutopilotModeSelector/helpers";
import { UnsupervisedConfirmDialog } from "@/app/(platform)/copilot/components/ChatInput/components/AutopilotModeSelector/UnsupervisedConfirmDialog";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useExpertModeSection } from "./useExpertModeSection";

interface Props {
  expert: Expert;
}

export function ExpertModeSection({ expert }: Props) {
  const isAutoModeEnabled = useGetFlag(Flag.COPILOT_AUTO_MODE);
  if (!isAutoModeEnabled) return null;
  return <ExpertModeOptions expert={expert} />;
}

function ExpertModeOptions({ expert }: Props) {
  const {
    mode,
    isPending,
    selectMode,
    isConfirmOpen,
    confirmUnsupervised,
    cancelUnsupervised,
  } = useExpertModeSection({ expert });

  return (
    <section
      aria-label={`${expert.name} approval mode`}
      className="rounded-xl border border-border-base bg-surface p-4"
    >
      <Text variant="large-medium">Approvals</Text>
      <Text variant="small" tone="secondary" className="mt-1">
        How every new web chat with {expert.name} starts. You can still change
        it per chat from the composer, and chats already open keep their own
        setting. Routines, scheduled runs and chats from Slack, Discord, Teams
        or Telegram are not affected.
      </Text>
      <fieldset className="mt-4 flex flex-col gap-2" disabled={isPending}>
        <legend className="sr-only">
          Approval mode for new chats with {expert.name}
        </legend>
        {AUTOPILOT_MODE_OPTIONS.map((option) => {
          const isSelected = option.value === mode;
          return (
            <label
              key={option.value}
              className={cn(
                // TODO(tokens): no fitting role for has-[:focus-visible]:outline-zinc-800 #3E3E43; #3E3E43 is foreground-emphasis (a text colour), action (a button fill)
                "flex cursor-pointer items-start gap-3 rounded-xl p-3 text-left transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-zinc-800",
                isSelected
                  ? // TODO(tokens): no fitting role for ring-zinc-800 #3E3E43; #3E3E43 is foreground-emphasis (a text colour), action (a button fill)
                    "bg-surface ring-2 ring-inset ring-zinc-800"
                  : "bg-surface-muted ring-1 ring-inset ring-border-base hover:bg-surface-hover",
                option.value === "unsupervised" &&
                  isSelected &&
                  // TODO(tokens): no exact role: ring-amber-600 #D97706
                  "bg-warning-surface ring-amber-600",
              )}
            >
              <input
                type="radio"
                name="expert-autopilot-mode"
                value={option.value}
                checked={isSelected}
                onChange={() => selectMode(option.value)}
                className="sr-only"
              />
              <Icon
                icon={option.icon}
                size={18}
                className="mt-0.5 shrink-0 text-foreground-body"
                aria-hidden="true"
              />
              <span className="flex flex-col gap-0.5">
                <Text variant="body-md-medium" as="span" tone="primary">
                  {option.label}
                  {option.value === DEFAULT_AUTOPILOT_MODE ? " (default)" : ""}
                </Text>
                <Text variant="small-tight" as="span" tone="muted">
                  {option.description}
                </Text>
              </span>
            </label>
          );
        })}
      </fieldset>
      <UnsupervisedConfirmDialog
        isOpen={isConfirmOpen}
        onConfirm={confirmUnsupervised}
        onCancel={cancelUnsupervised}
        title={`Run ${expert.name} unsupervised by default?`}
        description={`${expert.name} will not ask before anything in new chats. Edits, commands and actions outside the platform, like sending a message or an email, run without your approval unless you switch a chat back from the composer.`}
      />
    </section>
  );
}
