"use client";

import { CheckIcon, CopyIcon, ShareNetworkIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useShareChatDialog } from "./useShareChatDialog";

type Props = {
  sessionId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function ShareChatDialog({ sessionId, open, onOpenChange }: Props) {
  const state = useShareChatDialog({ sessionId, open });

  return (
    <Dialog
      title="Share this chat"
      controlled={{ isOpen: open, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="space-y-4">
          <div className="rounded-md border border-amber-200 bg-amber-50 p-3">
            <Text variant="small" className="text-amber-900">
              Anyone with the link will see this conversation. Don&apos;t share
              if it contains secrets you pasted, personal details, or
              credentials you wouldn&apos;t want public.
            </Text>
          </div>

          {/* Global execution-sharing toggle.  When ON, every agent
              run in this chat — past and future — is included in the
              share automatically.  Locked while shared so the toggle
              state always matches what the viewer sees. */}
          <div className="flex items-center justify-between gap-3 rounded border border-zinc-200 px-3 py-2.5">
            <div className="min-w-0">
              <Text variant="body" className="font-medium">
                Share agent runs in this chat
              </Text>
              <Text variant="small" className="text-zinc-500">
                Includes every run from this conversation, including ones that
                happen after you share.
              </Text>
            </div>
            <Switch
              checked={state.autoShareExecutions}
              onCheckedChange={state.setAutoShareExecutions}
              disabled={state.isShared || state.isLoadingState}
              aria-label="Share agent runs from this chat"
            />
          </div>

          {state.isShared && state.shareUrl && (
            <div className="space-y-2">
              <Text variant="small" className="font-medium">
                Share link
              </Text>
              <div className="flex items-center gap-2">
                <input
                  readOnly
                  value={state.shareUrl}
                  className="flex-1 rounded border border-zinc-200 bg-zinc-50 px-2 py-1.5 font-mono text-xs"
                />
                <Button
                  size="small"
                  variant="secondary"
                  onClick={state.copyShareUrl}
                  leftIcon={
                    state.copied ? (
                      <CheckIcon size={14} weight="bold" />
                    ) : (
                      <CopyIcon size={14} />
                    )
                  }
                >
                  {state.copied ? "Copied" : "Copy"}
                </Button>
              </div>
            </div>
          )}
        </div>
        <Dialog.Footer>
          {state.isShared ? (
            <Button
              variant="destructive"
              onClick={state.disable}
              loading={state.isDisabling}
            >
              Stop sharing
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={state.enable}
              loading={state.isEnabling}
              disabled={state.isLoadingState}
              leftIcon={<ShareNetworkIcon size={14} />}
            >
              Enable sharing
            </Button>
          )}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
