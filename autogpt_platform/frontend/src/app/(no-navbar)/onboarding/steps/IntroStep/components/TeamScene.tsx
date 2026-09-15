import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  AUTOPILOT_AVATAR,
  type AvatarConfig,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";
import { motion } from "framer-motion";
import { facing } from "../helpers";

interface Member {
  name: string;
  config: AvatarConfig;
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
    config: AUTOPILOT_AVATAR,
    status: "idle",
    size: 104,
    x: 210,
    y: 56,
  },
  {
    name: "Ops",
    config: { shape: "squircle", color: "mint", accessory: "headset" },
    status: "working",
    size: 84,
    x: 105,
    y: 176,
    parent: "Otto",
  },
  {
    name: "Research",
    config: { shape: "round", color: "sky", accessory: "glasses" },
    status: "thinking",
    size: 84,
    x: 210,
    y: 176,
    parent: "Otto",
  },
  {
    name: "Marketing",
    config: { shape: "bean", color: "coral", accessory: "bow" },
    status: "done",
    size: 84,
    x: 315,
    y: 176,
    parent: "Otto",
  },
  {
    name: "Finance",
    config: { shape: "wide", color: "amber", accessory: "badge" },
    status: "idle",
    size: 72,
    x: 52,
    y: 298,
    parent: "Ops",
  },
  {
    name: "Support",
    config: { shape: "dome", color: "plum", accessory: "flower" },
    status: "working",
    size: 72,
    x: 158,
    y: 298,
    parent: "Ops",
  },
  {
    name: "Sales",
    config: { shape: "round", color: "indigo", accessory: "star" },
    status: "idle",
    size: 72,
    x: 262,
    y: 298,
    parent: "Marketing",
  },
  {
    name: "Design",
    config: { shape: "squircle", color: "butter", accessory: "headband" },
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

function yawToward(member: Member): number {
  return ((WIDTH / 2 - member.x) / WIDTH) * 40;
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
              <BotAvatar
                config={member.config}
                status={member.status}
                size={member.size}
                poseOffset={facing(yawToward(member))}
                trackPointer={member.name === "Otto"}
                showBadge={false}
              />
            </motion.div>
          </foreignObject>
        ))}
      </svg>
    </div>
  );
}
