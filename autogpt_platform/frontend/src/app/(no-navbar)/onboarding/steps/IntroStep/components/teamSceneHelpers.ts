interface TeamMember {
  name: string;
  asset: string;
  x: number;
  y: number;
  size: number;
  parent?: string;
}

export const TEAM: TeamMember[] = [
  { name: "Otto", asset: "otto", x: 300, y: 48, size: 88 },
  { name: "Maria", asset: "maria", x: 150, y: 202, size: 80, parent: "otto" },
  { name: "Max", asset: "max", x: 300, y: 202, size: 80, parent: "otto" },
  { name: "Mina", asset: "mina", x: 450, y: 202, size: 80, parent: "otto" },
  {
    name: "Frankie",
    asset: "frankie",
    x: 75,
    y: 360,
    size: 72,
    parent: "maria",
  },
  { name: "Devon", asset: "devon", x: 225, y: 360, size: 72, parent: "maria" },
  { name: "Riley", asset: "riley", x: 375, y: 360, size: 72, parent: "mina" },
  { name: "Nadia", asset: "nadia", x: 525, y: 360, size: 72, parent: "mina" },
];

export const TEAM_PARENTS = TEAM.filter((member) =>
  TEAM.some((child) => child.parent === member.asset),
);

export function teamBranch(parent: TeamMember): string {
  const children = TEAM.filter((member) => member.parent === parent.asset);
  const start = parent.y + parent.size / 2 + 24;
  const top = Math.min(...children.map((child) => child.y - child.size / 2));
  const middle = (start + top) / 2;
  const left = children[0].x;
  const right = children[children.length - 1].x;
  const corner = Math.min(10, top - middle);
  const drops = children
    .slice(1, -1)
    .map((child) => `M ${child.x} ${middle} V ${top}`)
    .join(" ");

  return [
    `M ${parent.x} ${start} V ${middle}`,
    `M ${left} ${top} V ${middle + corner}`,
    `Q ${left} ${middle} ${left + corner} ${middle}`,
    `H ${right - corner}`,
    `Q ${right} ${middle} ${right} ${middle + corner} V ${top}`,
    drops,
  ].join(" ");
}
