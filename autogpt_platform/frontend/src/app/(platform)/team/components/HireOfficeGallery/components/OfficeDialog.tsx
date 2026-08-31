"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { HireOfficeResponse, OfficeTemplate } from "../api";
import { HiredRoster } from "./HiredRoster";
import { OfficePreviewList } from "./OfficePreviewList";

interface Props {
  template: OfficeTemplate | null;
  hireResult: HireOfficeResponse | null;
  isHiring: boolean;
  onHire: (template: OfficeTemplate) => void;
  onClose: () => void;
}

export function OfficeDialog({
  template,
  hireResult,
  isHiring,
  onHire,
  onClose,
}: Props) {
  if (!template) return null;

  return (
    <Dialog
      title={
        hireResult ? `${hireResult.office_name} is on board` : template.name
      }
      styling={{ maxWidth: "32rem" }}
      controlled={{
        isOpen: true,
        set: (next) => {
          if (!next) onClose();
        },
      }}
    >
      <Dialog.Content>
        {hireResult ? (
          <HiredRoster result={hireResult} onDone={onClose} />
        ) : (
          <div className="flex flex-col gap-4">
            <Text variant="body" className="text-zinc-600">
              {template.description}
            </Text>
            <OfficePreviewList experts={template.experts} />
            <div className="flex justify-end gap-2">
              <Button type="button" variant="secondary" onClick={onClose}>
                Cancel
              </Button>
              <Button
                type="button"
                variant="primary"
                loading={isHiring}
                disabled={isHiring}
                onClick={() => onHire(template)}
              >
                Hire office
              </Button>
            </div>
          </div>
        )}
      </Dialog.Content>
    </Dialog>
  );
}
