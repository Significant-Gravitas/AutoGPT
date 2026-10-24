"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { platformOption } from "../../helpers";
import { UsageGuide } from "../UsageGuide/UsageGuide";
import { CreateWorkloadForm } from "./components/CreateWorkloadForm";
import { PlatformPicker } from "./components/PlatformPicker";
import { useCreateWorkloadForm } from "./useCreateWorkloadForm";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateWorkloadDialog({ open, onOpenChange }: Props) {
  const {
    form,
    view,
    platform,
    canSubmit,
    created,
    isCreating,
    choosePlatform,
    handleSubmit,
    handleBack,
    handleClose,
  } = useCreateWorkloadForm({ onClose: () => onOpenChange(false) });

  const titles = {
    platform: "Where does your code run?",
    details: `Trust ${platformOption(platform).label}`,
    success: "Workload trusted. Here is how to use it",
  };

  return (
    <Dialog
      title={titles[view]}
      styling={{ maxWidth: "44rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (next) {
            onOpenChange(true);
            return;
          }
          if (isCreating) return;
          handleClose();
        },
      }}
    >
      <Dialog.Content>
        {view === "platform" && <PlatformPicker onPick={choosePlatform} />}
        {view === "details" && (
          <CreateWorkloadForm
            form={form}
            platform={platform}
            canSubmit={canSubmit}
            isPending={isCreating}
            onSubmit={handleSubmit}
            onBack={handleBack}
          />
        )}
        {view === "success" && created && (
          <div className="flex flex-col gap-5 px-1">
            <UsageGuide
              identity={created}
              platform={platform}
              showSiteSetup={platform === "website"}
            />
            <Button variant="primary" size="large" onClick={handleClose}>
              Done
            </Button>
          </div>
        )}
      </Dialog.Content>
    </Dialog>
  );
}
