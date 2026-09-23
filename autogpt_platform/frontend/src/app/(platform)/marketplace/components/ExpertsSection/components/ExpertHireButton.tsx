"use client";

import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { markHireStarted } from "@/services/experts/hire-timing";
import { useHireFlow } from "@/services/experts/useHireFlow";

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
      <Button as="NextLink" href={`/signup?next=${next}`} size="small">
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
      <Button size="small" loading={isHiring} onClick={handleHire}>
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
