export interface FilterOption {
  label: string;
  count: number;
  value: string;
}

interface Args {
  totalCount: number;
  agentsCount: number;
  creatorsCount: number;
  skillsCount?: number;
  expertsCount?: number;
}

export function getFilterOptions({
  totalCount,
  agentsCount,
  creatorsCount,
  skillsCount,
  expertsCount,
}: Args): FilterOption[] {
  return [
    { label: "All", count: totalCount, value: "all" },
    ...(expertsCount === undefined
      ? []
      : [{ label: "Experts", count: expertsCount, value: "experts" }]),
    { label: "Workflows", count: agentsCount, value: "agents" },
    ...(skillsCount === undefined
      ? []
      : [{ label: "Skills", count: skillsCount, value: "skills" }]),
    { label: "Creators", count: creatorsCount, value: "creators" },
  ];
}
