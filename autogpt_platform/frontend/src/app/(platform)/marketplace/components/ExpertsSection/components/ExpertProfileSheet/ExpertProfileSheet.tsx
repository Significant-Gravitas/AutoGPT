"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
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
  const { isHired, isHiring, hire } = useExpertProfileSheet(expert, onClose);
  const accent = expert ? getExpertAccent(expert.role) : null;

  const content =
    expert && accent ? (
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
          if (!open) onClose();
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
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>{content}</Dialog.Content>
    </Dialog>
  );
}
