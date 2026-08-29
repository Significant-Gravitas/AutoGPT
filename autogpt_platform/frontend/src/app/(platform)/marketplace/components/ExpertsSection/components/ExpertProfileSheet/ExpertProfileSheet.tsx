"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { getExpertAccent } from "../../helpers";
import { ExpertProfileContent } from "./ExpertProfileContent";
import { useExpertProfileSheet } from "./useExpertProfileSheet";

interface Props {
  expert: Expert | null;
  onClose: () => void;
  presentation?: "dialog" | "drawer";
}

export function ExpertProfileSheet({
  expert,
  onClose,
  presentation = "dialog",
}: Props) {
  const {
    isHired,
    isHiring,
    hire,
    hireResult,
    pickVoice,
    skipVoice,
    isSavingVoice,
    handleClose,
  } = useExpertProfileSheet(expert, onClose);
  const accent = expert ? getExpertAccent(expert.role) : null;

  const content = hireResult ? (
    <VoicePicker
      name={hireResult.expert.name}
      samples={expert?.voice_samples ?? []}
      onPick={pickVoice}
      onSkip={skipVoice}
      isSubmitting={isSavingVoice}
    />
  ) : expert && accent ? (
    <ExpertProfileContent
      expert={expert}
      accent={accent}
      isHired={isHired}
      isHiring={isHiring}
      onHire={hire}
    />
  ) : null;

  if (presentation === "drawer") {
    return (
      <Sheet
        open={expert !== null}
        onOpenChange={(open) => {
          if (!open) handleClose();
        }}
      >
        <SheetContent
          side="right"
          className="w-full overflow-y-auto sm:w-1/2 sm:max-w-none"
        >
          <SheetTitle className="sr-only">{expert?.name}</SheetTitle>
          {content}
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <Dialog
      styling={{ width: "640px" }}
      controlled={{
        isOpen: expert !== null,
        set: (open) => {
          if (!open) handleClose();
        },
      }}
    >
      <Dialog.Content>{content}</Dialog.Content>
    </Dialog>
  );
}
