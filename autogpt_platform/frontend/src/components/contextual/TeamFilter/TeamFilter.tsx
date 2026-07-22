"use client";

import { Select } from "@/components/atoms/Select/Select";
import type { SelectOption } from "@/components/atoms/Select/Select";
import { useOrgTeamStore } from "@/services/org-team/store";
import { TEAM_FILTER_ALL, TEAM_FILTER_ORG_HOME } from "./helpers";

interface Props {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  className?: string;
  wrapperClassName?: string;
}

// Team filter control for list headers. Options: All / Organization / each
// team. Renders nothing for solo users (no teams), keeping lists unchanged.
export function TeamFilter({
  value,
  onChange,
  label = "Team",
  className,
  wrapperClassName,
}: Props) {
  const teams = useOrgTeamStore((s) => s.teams);
  if (teams.length === 0) return null;

  const options: SelectOption[] = [
    { value: TEAM_FILTER_ALL, label: "All teams" },
    { value: TEAM_FILTER_ORG_HOME, label: "Organization" },
    ...teams.map((team) => ({ value: team.id, label: team.name })),
  ];

  return (
    <Select
      id="team-filter"
      label={label}
      hideLabel
      value={value}
      onValueChange={onChange}
      options={options}
      size="small"
      className={className}
      wrapperClassName={wrapperClassName}
    />
  );
}
