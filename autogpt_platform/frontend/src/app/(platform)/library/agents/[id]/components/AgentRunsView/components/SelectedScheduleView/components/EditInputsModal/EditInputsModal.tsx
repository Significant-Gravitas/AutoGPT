"use client";

import React from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { RunAgentInputs } from "../../../RunAgentInputs/RunAgentInputs";
import { useEditInputsModal } from "./useEditInputsModal";
import { PencilSimpleIcon } from "@phosphor-icons/react";

type Props = {
  agent: LibraryAgent;
  schedule: GraphExecutionJobInfo;
};

export function EditInputsModal({ agent, schedule }: Props) {
  const {
    isOpen,
    setIsOpen,
    inputFields,
    values,
    setValues,
    handleSave,
    isSaving,
  } = useEditInputsModal(agent, schedule);

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "32rem" }}
    >
      <Dialog.Trigger>
        <Button
          variant="ghost"
          size="small"
          className="absolute -right-2 -top-2"
        >
          <PencilSimpleIcon className="size-4" /> Edit inputs
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="h3">Edit inputs</Text>
          <div className="flex flex-col gap-4">
            {Object.entries(inputFields).map(([key, fieldSchema]) => (
              <div key={key} className="flex flex-col gap-1.5">
                <label className="text-sm font-medium">
                  {fieldSchema?.title || key}
                </label>
                <RunAgentInputs
                  schema={fieldSchema as any}
                  value={values[key]}
                  onChange={(v) => setValues((prev) => ({ ...prev, [key]: v }))}
                />
              </div>
            ))}
          </div>
        </div>
        <Dialog.Footer>
          <div className="flex w-full justify-end gap-2">
            <Button
              variant="secondary"
              size="small"
              onClick={() => setIsOpen(false)}
              className="min-w-32"
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              size="small"
              onClick={handleSave}
              loading={isSaving}
              className="min-w-32"
            >
              {isSaving ? "Saving…" : "Save"}
            </Button>
          </div>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
