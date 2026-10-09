import Image from "next/image";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { TEAM, TEAM_PARENTS, teamBranch } from "./teamSceneHelpers";

export function TeamScene() {
  return (
    <div
      role="group"
      className="flex h-full items-center justify-center"
      aria-label="Otto and your team of AI Experts"
    >
      <svg viewBox="0 0 600 420" className="h-auto w-full max-w-[600px]">
        {TEAM_PARENTS.map((member) => (
          <path
            key={member.asset}
            d={teamBranch(member)}
            aria-hidden="true"
            fill="none"
            className="stroke-border"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {TEAM.map((member) => (
          <TeamNode key={member.asset} member={member} />
        ))}
      </svg>
    </div>
  );
}

interface Props {
  member: (typeof TEAM)[number];
}

function TeamNode({ member }: Props) {
  return (
    <g>
      <foreignObject
        x={member.x - member.size / 2}
        y={member.y - member.size / 2}
        width={member.size}
        height={member.size}
      >
        {member.asset === "otto" ? (
          <AutopilotAvatar size={member.size} transparent />
        ) : (
          <Image
            src={`/experts/transparent/${member.asset}.webp`}
            alt={`${member.name}, AI Expert`}
            width={member.size}
            height={member.size}
            sizes={`${member.size}px`}
            className="h-full w-full object-contain"
          />
        )}
      </foreignObject>
      <foreignObject
        x={member.x - 72}
        y={member.y + member.size / 2 + 4}
        width={144}
        height={20}
      >
        <div className="text-center text-xs leading-4">
          <span className="block font-medium text-foreground">
            {member.name}
          </span>
        </div>
      </foreignObject>
    </g>
  );
}
