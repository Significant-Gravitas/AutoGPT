"use client";

import { Select } from "@/components/atoms/Select/Select";
import { useTeamPicker } from "./useTeamPicker";

interface Props {
  surfaceKey: string;
  value: string | null;
  onChange: (teamId: string | null) => void;
  label?: string;
  hideLabel?: boolean;
  className?: string;
  wrapperClassName?: string;
}

// Explicit team ownership picker for create flows. Lists "Organization"
// (org-home) plus the user's teams. Renders nothing for solo users (no teams),
// so those surfaces look unchanged.
export function TeamPicker({
  surfaceKey,
  value,
  onChange,
  label = "Team",
  hideLabel,
  className,
  wrapperClassName,
}: Props) {
  const { hasTeams, options, selectValue, handleChange } = useTeamPicker({
    value,
    onChange,
  });

  if (!hasTeams) return null;

  return (
    <Select
      id={`team-picker-${surfaceKey}`}
      label={label}
      hideLabel={hideLabel}
      value={selectValue}
      onValueChange={handleChange}
      options={options}
      size="small"
      className={className}
      wrapperClassName={wrapperClassName}
    />
  );
}
