"use client";

import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import {
  BAN_DURATIONS,
  type BanDuration,
  formatDateTime,
  formatTimeUntil,
  getBanEnd,
} from "../../helpers";

const DURATION_OPTIONS = BAN_DURATIONS.map(({ value, label }) => ({
  value,
  label,
}));

interface Props {
  idPrefix: string;
  reason: string;
  onReasonChange: (reason: string) => void;
  duration: BanDuration;
  onDurationChange: (duration: BanDuration) => void;
}

export function ReasonFields({
  idPrefix,
  reason,
  onReasonChange,
  duration,
  onDurationChange,
}: Props) {
  const end = getBanEnd(duration);

  return (
    <>
      <Input
        id={`${idPrefix}-reason`}
        label="Reason (required)"
        type="textarea"
        rows={3}
        placeholder="What they did"
        value={reason}
        onChange={(event) => onReasonChange(event.target.value)}
        required
        wrapperClassName="mb-0"
      />
      <div className="flex flex-col gap-1">
        <Select
          id={`${idPrefix}-duration`}
          label="Duration"
          value={duration}
          onValueChange={(value) => onDurationChange(value as BanDuration)}
          options={DURATION_OPTIONS}
          wrapperClassName="mb-0"
        />
        <Text variant="small" className="text-foreground-tertiary">
          {end
            ? `Ends ${formatDateTime(end)} (${formatTimeUntil(end)})`
            : "Doesn't end on its own."}
        </Text>
      </div>
    </>
  );
}
