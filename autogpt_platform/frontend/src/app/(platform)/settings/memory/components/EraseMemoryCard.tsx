"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";

interface Props {
  scopeName: string;
  memoryCount: number | null;
  isErasing: boolean;
  onErase: () => Promise<boolean>;
}

export function EraseMemoryCard({
  scopeName,
  memoryCount,
  isErasing,
  onErase,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const isConfirmed = confirmText.trim() === scopeName;

  async function handleErase() {
    if (!isConfirmed) return;
    const erased = await onErase();
    if (erased) setIsOpen(false);
  }

  return (
    <div className="flex flex-col gap-3 rounded-[18px] border border-red-200 bg-white px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)] sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="body-medium" as="span" className="text-red-700">
          Erase {scopeName}&apos;s memory
        </Text>
        <Text variant="small" as="span" className="text-zinc-500">
          Permanently delete everything this memory holds. This can&apos;t be
          undone.
        </Text>
      </div>
      <Dialog
        title={`Erase all of ${scopeName}'s memory`}
        styling={{ maxWidth: "440px" }}
        controlled={{
          isOpen,
          set: (open) => {
            setIsOpen(open);
            if (open) setConfirmText("");
          },
        }}
      >
        <Dialog.Trigger>
          <Button variant="destructive" size="small">
            Erase memory
          </Button>
        </Dialog.Trigger>
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="small" as="p" className="text-zinc-600">
              {memoryCount !== null
                ? `Every memory in this scope will be permanently deleted — ${memoryCount} ${memoryCount === 1 ? "memory" : "memories"}, raw conversations included. Other scopes are not affected.`
                : "Every memory in this scope will be permanently deleted, raw conversations included. Other scopes are not affected."}{" "}
              Type <strong>{scopeName}</strong> to confirm.
            </Text>
            <Input
              id="erase-memory-confirm"
              label={`Type ${scopeName} to confirm`}
              hideLabel
              placeholder={scopeName}
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
            />
            <div className="flex justify-end gap-2">
              <Button
                variant="secondary"
                size="small"
                onClick={() => setIsOpen(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                size="small"
                disabled={!isConfirmed}
                loading={isErasing}
                onClick={handleErase}
              >
                Erase everything
              </Button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
