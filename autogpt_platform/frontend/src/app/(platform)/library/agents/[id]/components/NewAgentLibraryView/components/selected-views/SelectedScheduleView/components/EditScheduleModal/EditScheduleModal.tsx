"use client";

import React from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Input } from "@/components/atoms/Input/Input";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";
import { Select } from "@/components/atoms/Select/Select";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useEditScheduleModal } from "./useEditScheduleModal";
import { PencilSimpleIcon } from "@phosphor-icons/react";

type Props = {
  graphId: string;
  schedule: GraphExecutionJobInfo;
};

export function EditScheduleModal({ graphId, schedule }: Props) {
  const {
    isOpen,
    setIsOpen,
    name,
    setName,
    repeat,
    setRepeat,
    selectedDays,
    setSelectedDays,
    time,
    setTime,
    errors,
    repeatOptions,
    dayItems,
    mutateAsync,
    isPending,
  } = useEditScheduleModal(graphId, schedule);

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "22rem" }}
    >
      <Dialog.Trigger>
        <Button
          variant="ghost"
          size="small"
          className="absolute -right-2 -top-2"
        >
          <PencilSimpleIcon className="size-4" /> Edit schedule
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-6">
          <Text variant="h3">Edit schedule</Text>
          <Input
            id="schedule-name"
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            error={errors.scheduleName}
          />
          <Select
            id="repeat"
            label="Repeats"
            value={repeat}
            onValueChange={setRepeat}
            options={repeatOptions}
          />
          {repeat === "weekly" && (
            <MultiToggle
              items={dayItems}
              selectedValues={selectedDays}
              onChange={setSelectedDays}
              aria-label="Select days"
            />
          )}
          <Input
            id="schedule-time"
            label="At"
            value={time}
            onChange={(e) => setTime(e.target.value.trim())}
            placeholder="00:00"
            error={errors.time}
          />
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
              onClick={() => mutateAsync()}
              loading={isPending}
              className="min-w-32"
            >
              {isPending ? "Saving…" : "Save"}
            </Button>
          </div>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
