"use client";

import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { platformOf } from "../../helpers";
import { UsageGuide } from "../UsageGuide/UsageGuide";

interface Props {
  identity: WorkloadIdentityInfo;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function UsageDialog({ identity, open, onOpenChange }: Props) {
  return (
    <Dialog
      title={`How to use ${identity.name}`}
      styling={{ maxWidth: "44rem" }}
      controlled={{ isOpen: open, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="px-1">
          <UsageGuide identity={identity} platform={platformOf(identity)} />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
