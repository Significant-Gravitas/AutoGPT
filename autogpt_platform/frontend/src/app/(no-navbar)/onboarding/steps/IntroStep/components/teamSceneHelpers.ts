// The team picture is laid out on a 600-unit-wide, 396px-tall stage that stretches to the
// screen's width, while every vertical position is in pixels: names and roles
// are text at a fixed size, so the rows keep the room their labels need on a
// phone instead of shrinking with the drawing.
const STAGE_WIDTH = 600;

// A name over a role, from the top of the label to its last line.
export const LABEL_GAP = 6;
export const LABEL_HEIGHT = 40;
// Space between a connector and the label above it.
const CONNECTOR_GAP = 8;

export interface TeamMember {
  name: string;
  role: string;
  asset: string;
  /** Centre, in stage units (0-600). */
  x: number;
  /** Top of the picture, in pixels. */
  top: number;
  /** Picture size in pixels on a screen wide enough for the whole stage. */
  size: number;
  parent?: string;
}

// Otto leads three heads of department, and each side of the tree is a team
// whose work belongs together: research and support under marketing,
// operations and security under finance.
export const TEAM: TeamMember[] = [
  { name: "Otto", role: "Head of AI", asset: "otto", x: 300, top: 0, size: 88 },
  {
    name: "Maria",
    role: "Marketing",
    asset: "maria",
    x: 150,
    top: 128,
    size: 76,
    parent: "otto",
  },
  {
    name: "Max",
    role: "Sales",
    asset: "max",
    x: 300,
    top: 128,
    size: 76,
    parent: "otto",
  },
  {
    name: "Mina",
    role: "Finance",
    asset: "mina",
    x: 450,
    top: 128,
    size: 76,
    parent: "otto",
  },
  {
    name: "Nadia",
    role: "Research",
    asset: "nadia",
    x: 75,
    top: 282,
    size: 68,
    parent: "maria",
  },
  {
    name: "Riley",
    role: "Support",
    asset: "riley",
    x: 225,
    top: 282,
    size: 68,
    parent: "maria",
  },
  {
    name: "Frankie",
    role: "Operations",
    asset: "frankie",
    x: 375,
    top: 282,
    size: 68,
    parent: "mina",
  },
  {
    name: "Devon",
    role: "Security",
    asset: "devon",
    x: 525,
    top: 282,
    size: 68,
    parent: "mina",
  },
];

export const TEAM_PARENTS = TEAM.filter((member) =>
  TEAM.some((child) => child.parent === member.asset),
);

export function isLead(member: TeamMember) {
  return !member.parent;
}

// Where a parent's connector starts: under the picture for Otto, whose label
// sits beside him, and under the label for everyone else.
function connectorStart(parent: TeamMember) {
  const bottom = parent.top + parent.size;
  if (isLead(parent)) return bottom + CONNECTOR_GAP;
  return bottom + LABEL_GAP + LABEL_HEIGHT + CONNECTOR_GAP;
}

/** One parent's branch: a stem down from the parent to a bar halfway to
 *  the children, and the bar's drops into each child. Horizontal positions
 *  are stage units, vertical ones pixels at full picture size. */
export function teamBranch(parent: TeamMember) {
  const children = TEAM.filter((member) => member.parent === parent.asset);
  const start = connectorStart(parent);
  const end = Math.min(...children.map((child) => child.top));
  return {
    x: parent.x,
    start,
    end,
    left: children[0].x,
    right: children[children.length - 1].x,
    innerDrops: children.slice(1, -1).map((child) => child.x),
  };
}

export function stagePercent(x: number) {
  return `${(x / STAGE_WIDTH) * 100}%`;
}

/** A picture's rendered size: full size down to a 360px-wide stage, then
 *  shrinking with it so neighbours never touch on the narrowest phones. */
export function fluidSize(size: number) {
  return `min(${size}px, ${size / 3.6}cqw)`;
}

// The order the team arrives in, in seconds: Otto, his branch, the heads of
// department left to right, their branches, then the bottom row.
export function revealDelay(member: TeamMember) {
  if (isLead(member)) return 0.1;
  const row = TEAM.filter((other) => other.top === member.top);
  const rowStart = member.parent === "otto" ? 0.45 : 0.8;
  return rowStart + row.indexOf(member) * 0.07;
}

export function branchDelay(parent: TeamMember) {
  return isLead(parent) ? 0.3 : 0.65;
}
