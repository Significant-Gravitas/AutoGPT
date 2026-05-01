"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useRouter } from "next/navigation";
import { formatResetTime } from "../usageHelpers";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  resetsAt?: string | Date | null;
}

export function RateLimitResetDialog({ isOpen, onClose, resetsAt }: Props) {
  const router = useRouter();
  const resetTimeLabel = resetsAt ? formatResetTime(resetsAt) : null;

  return (
    <Dialog
      title="Daily AutoPilot limit reached"
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          You&apos;ve reached your daily usage limit.
          {resetTimeLabel && resetTimeLabel !== "now"
            ? ` Resets ${resetTimeLabel}.`
            : ""}{" "}
          You can still browse, edit agents, and view results &mdash; or upgrade
          your plan.
        </Text>
        <Dialog.Footer className="!justify-center">
          <Button variant="secondary" onClick={onClose}>
            Wait for reset
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              onClose();
              router.push("/settings/billing");
            }}
          >
            Go to billing
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
