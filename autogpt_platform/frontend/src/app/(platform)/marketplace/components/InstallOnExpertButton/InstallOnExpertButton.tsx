"use client";

import { Button } from "@/components/atoms/Button/Button";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { useInstallOnExpertButton } from "./useInstallOnExpertButton";

interface Props {
  storeListingVersionId: string;
}

export function InstallOnExpertButton({ storeListingVersionId }: Props) {
  const { canInstall, pickerOpen, openPicker, closePicker } =
    useInstallOnExpertButton();

  if (!canInstall) return null;

  return (
    <>
      <Button variant="secondary" onClick={openPicker}>
        Install on Expert…
      </Button>
      <InstallWorkflowPicker
        mode="pick-expert"
        storeListingVersionId={storeListingVersionId}
        open={pickerOpen}
        onClose={closePicker}
      />
    </>
  );
}
