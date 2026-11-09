"use client";

import {
  Alert01Icon,
  Calendar03Icon,
  ConferenceIcon,
  PauseIcon,
} from "@hugeicons/core-free-icons";
import type { TeamFilter } from "../../helpers";
import { PillFilterMenu } from "../PillFilterMenu";

const OPTIONS = [
  { value: "all", label: "All experts", icon: ConferenceIcon },
  { value: "scheduled", label: "Scheduled", icon: Calendar03Icon },
  { value: "needs-setup", label: "Needs setup", icon: Alert01Icon },
  { value: "paused", label: "Paused", icon: PauseIcon },
] as const;

interface Props {
  value: TeamFilter;
  onChange: (next: TeamFilter) => void;
}

export function TeamFilterMenu({ value, onChange }: Props) {
  return (
    <PillFilterMenu
      label="Filter experts"
      value={value}
      defaultValue="all"
      options={OPTIONS}
      onChange={onChange}
    />
  );
}
