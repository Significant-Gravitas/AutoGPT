"use client";

import { Select } from "@/components/atoms/Select/Select";
import { useState } from "react";
import { RESET_SCOPE_OPTIONS, type ResetScope } from "../helpers";
import { ResetUsageDialog } from "./ResetUsageDialog";

interface Props {
  dailyUsed: number;
  weeklyUsed: number;
  isResetting: boolean;
  onReset: (scope: ResetScope, onSuccess: () => void) => void;
}

export function ResetControls({
  dailyUsed,
  weeklyUsed,
  isResetting,
  onReset,
}: Props) {
  const [scope, setScope] = useState<ResetScope>("daily");

  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <Select
        id="reset-scope"
        label="What to reset"
        hideLabel
        size="small"
        value={scope}
        onValueChange={(value) => setScope(value as ResetScope)}
        options={RESET_SCOPE_OPTIONS}
        wrapperClassName="mb-0 w-[220px]"
      />
      <ResetUsageDialog
        scope={scope}
        nothingToReset={
          dailyUsed === 0 && (scope === "daily" || weeklyUsed === 0)
        }
        isResetting={isResetting}
        onConfirm={onReset}
      />
    </div>
  );
}
