"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { SparkleIcon } from "@phosphor-icons/react/dist/ssr";

interface Props {
  open: boolean;
  onClose: () => void;
  onReplay: () => void;
}

export function TourUpsellModal({ open, onClose, onReplay }: Props) {
  return (
    <Dialog
      title="Ready to build your own?"
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-6">
          <div className="flex items-start gap-3">
            <SparkleIcon size={24} weight="fill" className="text-violet-600" />
            <Text variant="body" className="text-zinc-600">
              Spin up your own AI agents in minutes. Sign up free to start
              building, no credit card required.
            </Text>
          </div>

          <div className="flex flex-col gap-2 sm:flex-row sm:justify-end">
            <Button variant="ghost" size="small" onClick={onReplay}>
              Replay demo
            </Button>
            <Button
              as="NextLink"
              href="/pricing"
              variant="secondary"
              size="small"
            >
              See pricing
            </Button>
            <Button as="NextLink" href="/signup" variant="primary" size="small">
              Sign up free
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
