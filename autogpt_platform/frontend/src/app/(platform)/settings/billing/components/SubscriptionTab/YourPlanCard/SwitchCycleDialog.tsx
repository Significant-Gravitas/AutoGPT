"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  targetCycle: "monthly" | "yearly";
  title?: string;
  body: { label?: string; text: string }[];
  isSaving: boolean;
  onConfirm: () => void;
}

export function SwitchCycleDialog({
  isOpen,
  onOpenChange,
  targetCycle,
  title,
  body,
  isSaving,
  onConfirm,
}: Props) {
  const targetLabel = targetCycle === "yearly" ? "Yearly" : "Monthly";

  return (
    <Dialog
      title={title ?? `Switch billing to ${targetLabel}?`}
      styling={{ maxWidth: "440px" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-2">
          {body.map((line, index) => (
            <div
              key={`${index}-${line.label ?? ""}-${line.text}`}
              className="flex flex-wrap gap-1"
            >
              {line.label ? (
                <Text
                  variant="body"
                  as="span"
                  className="font-semibold text-zinc-700"
                >
                  {line.label}
                </Text>
              ) : null}
              <Text variant="body" as="span" className="text-zinc-700">
                {line.text}
              </Text>
            </div>
          ))}
        </div>

        <Dialog.Footer>
          <Button
            type="button"
            variant="ghost"
            size="small"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="primary"
            size="small"
            onClick={onConfirm}
            disabled={isSaving}
            loading={isSaving}
          >
            Switch to {targetLabel}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
