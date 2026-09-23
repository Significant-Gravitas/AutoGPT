import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/status";
import {
  notionConfigForName,
  type NotionColorId,
} from "@/components/molecules/NotionAvatar/helpers";
import { NotionAvatarImage } from "@/components/molecules/NotionAvatar/NotionAvatarImage";
import { motion } from "framer-motion";

interface Member {
  name: string;
  /** Otto keeps his own drawing; everyone else gets a name-seeded face. */
  color?: NotionColorId;
  status: AvatarStatus;
  size: number;
  x: number;
  y: number;
  parent?: string;
}

const WIDTH = 420;
const HEIGHT = 340;
const CORNER = 10;

const TEAM: Member[] = [
  {
    name: "Otto",
    status: "idle",
    size: 104,
    x: 210,
    y: 56,
  },
  {
    name: "Ops",
    color: "emerald",
    status: "working",
    size: 84,
    x: 105,
    y: 176,
    parent: "Otto",
  },
  {
    name: "Research",
    color: "sky",
    status: "thinking",
    size: 84,
    x: 210,
    y: 176,
    parent: "Otto",
  },
  {
    name: "Marketing",
    color: "orange",
    status: "done",
    size: 84,
    x: 315,
    y: 176,
    parent: "Otto",
  },
  {
    name: "Finance",
    color: "amber",
    status: "idle",
    size: 72,
    x: 52,
    y: 298,
    parent: "Ops",
  },
  {
    name: "Support",
    color: "rose",
    status: "working",
    size: 72,
    x: 158,
    y: 298,
    parent: "Ops",
  },
  {
    name: "Sales",
    color: "indigo",
    status: "idle",
    size: 72,
    x: 262,
    y: 298,
    parent: "Marketing",
  },
  {
    name: "Design",
    color: "yellow",
    status: "thinking",
    size: 72,
    x: 368,
    y: 298,
    parent: "Marketing",
  },
];

const BY_NAME = Object.fromEntries(TEAM.map((m) => [m.name, m]));
const PARENTS = TEAM.filter((m) => TEAM.some((c) => c.parent === m.name));

function depth(member: Member): number {
  return member.parent ? depth(BY_NAME[member.parent]) + 1 : 0;
}

function branch(parent: Member): string {
  const kids = TEAM.filter((c) => c.parent === parent.name);
  const stemStart = parent.y + parent.size / 2;
  const kidTop = Math.min(...kids.map((k) => k.y - k.size / 2));
  const bus = (stemStart + kidTop) / 2;
  const stem = `M ${parent.x} ${stemStart} V ${bus}`;
  const left = Math.min(...kids.map((k) => k.x));
  const right = Math.max(...kids.map((k) => k.x));
  const drops = kids
    .filter((k) => k.x !== left && k.x !== right)
    .map((k) => `M ${k.x} ${bus} V ${k.y - k.size / 2}`)
    .join(" ");
  const rail = [
    `M ${left} ${kidTop} V ${bus + CORNER}`,
    `Q ${left} ${bus} ${left + CORNER} ${bus}`,
    `H ${right - CORNER}`,
    `Q ${right} ${bus} ${right} ${bus + CORNER}`,
    `V ${kidTop}`,
  ].join(" ");
  return `${stem} ${rail} ${drops}`;
}

const EASE = [0.22, 1, 0.36, 1] as const;

export function TeamScene() {
  return (
    <div className="flex h-full items-center justify-center">
      <svg
        role="img"
        aria-label="Otto and your team of AI experts"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full max-w-[420px]"
      >
        {PARENTS.map((p) => (
          <motion.path
            key={p.name}
            d={branch(p)}
            fill="none"
            className="stroke-zinc-300"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{
              delay: 0.25 + depth(p) * 0.3,
              duration: 0.7,
              ease: EASE,
            }}
          />
        ))}
        {TEAM.map((member) => (
          <foreignObject
            key={member.name}
            x={member.x - member.size / 2}
            y={member.y - member.size / 2}
            width={member.size}
            height={member.size}
            overflow="visible"
          >
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{
                delay: 0.1 + depth(member) * 0.3,
                duration: 0.6,
                ease: EASE,
              }}
            >
              {member.color ? (
                <NotionAvatarImage
                  config={{
                    ...notionConfigForName(member.name),
                    color: member.color,
                  }}
                  status={member.status}
                  size={member.size}
                  title={member.name}
                />
              ) : (
                <AutopilotAvatar size={member.size} />
              )}
            </motion.div>
          </foreignObject>
        ))}
      </svg>
    </div>
  );
}
