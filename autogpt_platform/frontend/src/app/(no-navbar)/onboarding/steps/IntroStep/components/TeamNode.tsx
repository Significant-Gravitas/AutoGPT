import { Text } from "@/components/atoms/Text/Text";
import { motion } from "framer-motion";
import Image from "next/image";
import type { CSSProperties } from "react";
import {
  fluidSize,
  isLead,
  revealDelay,
  stagePercent,
  type TeamMember,
} from "./teamSceneHelpers";

const EASE = [0.22, 1, 0.36, 1] as const;

interface Props {
  member: TeamMember;
  animate: boolean;
}

export function TeamNode({ member, animate }: Props) {
  const lead = isLead(member);
  const position = {
    "--x": stagePercent(member.x),
    "--top": `${member.top}px`,
    "--size": fluidSize(member.size),
  } as CSSProperties;

  // The list item is placed and centred on its point; the motion inside it
  // owns the transform, so the two never fight over it.
  return (
    <li
      style={position}
      className="absolute top-(--top) left-(--x) -translate-x-1/2"
    >
      <motion.div
        className="flex flex-col items-center"
        initial={animate ? { opacity: 0, y: 10, scale: 0.92 } : false}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ delay: revealDelay(member), duration: 0.55, ease: EASE }}
      >
        <div className="relative size-(--size)">
          {lead && (
            <div
              aria-hidden="true"
              className="absolute -inset-4 rounded-full bg-brand/15 blur-2xl"
            />
          )}
          <div
            aria-hidden="true"
            className="absolute inset-x-[22%] bottom-0 h-2 rounded-full bg-scrim/10 blur-sm"
          />
          <Image
            src={`/experts/transparent/${member.asset}.webp`}
            alt=""
            width={member.size}
            height={member.size}
            sizes={`${member.size}px`}
            priority
            className="relative size-full object-contain"
          />
        </div>
        <span
          className={
            lead
              ? "absolute top-1/2 left-full ml-3 -translate-y-1/2"
              : "mt-1.5 text-center"
          }
        >
          <Text
            as="span"
            variant="body-medium"
            className="block whitespace-nowrap"
          >
            {member.name}
          </Text>
          <Text
            as="span"
            variant="small"
            tone={lead ? "accent" : "secondary"}
            className="block whitespace-nowrap"
          >
            {member.role}
          </Text>
        </span>
      </motion.div>
    </li>
  );
}
