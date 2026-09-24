"use client";

import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { markHireStarted } from "@/services/experts/hire-timing";
import { useHireFlow } from "@/services/experts/useHireFlow";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { AddTeamIcon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";

// The atom sizes a lone action; on a card it is a label with a glyph.
const TEXT_BUTTON = "min-w-0 gap-1.5 px-2 text-base";
const HIRE_ICON = <Icon icon={AddTeamIcon} size={20} aria-hidden />;

interface Props {
  expert: ExpertTemplate;
  isHired: boolean;
}

/** The card's corner: Hire, or the badge once the expert is on the team.
 *
 *  One component for both, deliberately. Hiring refetches the roster, which
 *  flips `isHired` while the flow is still mid-air; if the badge replaced the
 *  button as a separate component, the flow's state — the voice picker, the
 *  hand-off to the expert's first thread — would unmount with it. Here the
 *  flow runs the expert page's own hook and outlives the swap. */
export function ExpertHireControl({ expert, isHired }: Props) {
  const isHireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  if (!isHireExpertsEnabled) return null;

  return <EnabledExpertHireControl expert={expert} isHired={isHired} />;
}

function EnabledExpertHireControl({ expert, isHired }: Props) {
  const { isLoggedIn } = useAuth();
  const {
    hire,
    isHiring,
    isVoicePickOpen,
    hireResult,
    pickVoice,
    skipVoice,
    dismissVoicePick,
    isSavingVoice,
  } = useHireFlow(expert);
  const isInFlight = isHiring || isVoicePickOpen || hireResult !== null;

  // The clock starts on the click, not on the request: the flow finishes in
  // a dialog, and sometimes on another page entirely.
  function handleHire() {
    markHireStarted(expert.id);
    trackFunnel("hire_started", { template_id: expert.id });
    hire();
  }

  let control;
  if (isHired && !isInFlight) {
    control = (
      <Badge
        variant="success"
        className="rounded-full px-2.5 py-1 shadow-[0_1px_2px_rgba(16,24,40,0.05)]"
      >
        <Icon icon={CheckmarkCircle02Icon} size={14} aria-hidden />
        On your team
      </Badge>
    );
  } else if (!isLoggedIn) {
    const next = encodeURIComponent(`/marketplace/experts/${expert.id}`);
    control = (
      <Button
        as="NextLink"
        href={`/signup?next=${next}`}
        variant="ghost"
        size="small"
        leftIcon={HIRE_ICON}
        className={TEXT_BUTTON}
      >
        Hire
      </Button>
    );
  } else {
    control = (
      <Button
        variant="ghost"
        size="small"
        loading={isHiring}
        leftIcon={HIRE_ICON}
        onClick={handleHire}
        className={TEXT_BUTTON}
      >
        Hire
      </Button>
    );
  }

  return (
    <>
      {control}
      <Dialog
        styling={{ width: "640px" }}
        controlled={{
          isOpen: isVoicePickOpen,
          set: (open) => {
            if (!open) dismissVoicePick();
          },
        }}
      >
        <Dialog.Content>
          {isVoicePickOpen && hireResult ? (
            <VoicePicker
              name={hireResult.expert.name}
              samples={expert.voice_samples ?? []}
              onPick={pickVoice}
              onSkip={skipVoice}
              isSubmitting={isSavingVoice}
            />
          ) : null}
        </Dialog.Content>
      </Dialog>
    </>
  );
}
