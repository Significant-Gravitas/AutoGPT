"use client";

import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { markHireStarted } from "@/services/experts/hire-timing";
import { useHireFlow } from "@/services/experts/useHireFlow";
import { AddTeamIcon } from "@hugeicons/core-free-icons";

// The atom sizes a lone action; on a card it is a label with a glyph.
const TEXT_BUTTON = "min-w-0 px-2";

interface Props {
  expert: ExpertTemplate;
}

/** Hiring straight off the shelf. It runs the expert page's own flow, so a
 *  hire from a card celebrates, picks a voice and lands the user in the same
 *  place as one from the profile. */
export function ExpertHireButton({ expert }: Props) {
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

  if (!isLoggedIn) {
    const next = encodeURIComponent(`/marketplace/experts/${expert.id}`);
    return (
      <Button
        as="NextLink"
        href={`/signup?next=${next}`}
        variant="ghost"
        size="small"
        leadingIcon={AddTeamIcon}
        className={TEXT_BUTTON}
      >
        Hire
      </Button>
    );
  }

  // The clock starts on the click, not on the request: the flow finishes in
  // a dialog, and sometimes on another page entirely.
  function handleHire() {
    markHireStarted(expert.id);
    trackFunnel("hire_started", { template_id: expert.id });
    hire();
  }

  return (
    <>
      <Button
        variant="ghost"
        size="small"
        loading={isHiring}
        leadingIcon={AddTeamIcon}
        onClick={handleHire}
        className={TEXT_BUTTON}
      >
        Hire
      </Button>
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
