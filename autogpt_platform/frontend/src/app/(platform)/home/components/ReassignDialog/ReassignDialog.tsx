"use client";

import { ExpertAvatar } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/ExpertAvatar/ExpertAvatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { Card } from "../../board-helpers";
import { useReassignDialog } from "./useReassignDialog";

interface Props {
  card: Card | null;
  onClose: () => void;
  onReassigned: (newSessionId: string) => void;
}

export function ReassignDialog({ card, onClose, onReassigned }: Props) {
  const {
    options,
    selected,
    isPending,
    select,
    handleOpenChange,
    handleConfirm,
  } = useReassignDialog({ card, onClose, onReassigned });

  return (
    <Dialog
      title="Reassign task"
      styling={{ maxWidth: "28rem" }}
      controlled={{ isOpen: card !== null, set: handleOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          {options.length > 0 ? (
            <fieldset className="flex flex-col gap-1">
              <legend className="sr-only">Hand this task to</legend>
              {options.map((option) => (
                <label
                  key={option.id ?? "otto"}
                  className="flex cursor-pointer items-center gap-3 rounded-lg border border-transparent px-3 py-2 transition-colors has-[:checked]:border-zinc-300 has-[:checked]:bg-zinc-50 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-zinc-800 hover:bg-zinc-50"
                >
                  <input
                    type="radio"
                    name="reassign-to"
                    className="sr-only"
                    checked={selected !== null && selected.id === option.id}
                    onChange={() => select(option)}
                  />
                  <span aria-hidden className="flex-none">
                    <ExpertAvatar
                      name={option.name}
                      avatarUrl={option.avatarUrl}
                      color={option.color}
                      isAutopilot={option.id === null}
                      size="sm"
                    />
                  </span>
                  <span className="text-sm font-medium text-zinc-900">
                    {option.name}
                  </span>
                </label>
              ))}
            </fieldset>
          ) : (
            <Text variant="body" className="text-zinc-700">
              Hire an expert to hand this task to.
            </Text>
          )}
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="secondary" disabled={isPending} onClick={onClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={!selected}
              loading={isPending}
              onClick={handleConfirm}
            >
              Reassign
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
