import type { SelectOption } from "@/components/atoms/Select/Select";
import { useOrgTeamStore } from "@/services/org-team/store";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ORG_HOME_OPTION_VALUE } from "./helpers";

interface Params {
  value: string | null;
  onChange: (teamId: string | null) => void;
}

export function useTeamPicker({ value, onChange }: Params) {
  const enabled = useGetFlag(Flag.SHOW_ORG_SETTINGS);
  const allTeams = useOrgTeamStore((s) => s.teams);
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const teams = allTeams.filter((team) => team.orgId === activeOrgID);
  const hasTeams = enabled && teams.length > 0;

  const options: SelectOption[] = [
    { value: ORG_HOME_OPTION_VALUE, label: "Organization" },
    ...teams.map((team) => ({ value: team.id, label: team.name })),
  ];

  const knownIds = new Set(teams.map((team) => team.id));
  const selectValue =
    value && knownIds.has(value) ? value : ORG_HOME_OPTION_VALUE;

  function handleChange(next: string) {
    onChange(next === ORG_HOME_OPTION_VALUE ? null : next);
  }

  return { hasTeams, options, selectValue, handleChange };
}
