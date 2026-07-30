"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  motion,
  useReducedMotion,
  useTransform,
  type MotionValue,
} from "framer-motion";
import Image from "next/image";
import type { Persona } from "../../personas";
import { DIAL_RADIUS, DIAL_STEP, stepsFromBottom } from "./helpers";

interface Props {
  persona: Persona;
  index: number;
  isSelected: boolean;
  entranceDelay: number;
  rotation: MotionValue<number>;
  onPick: (index: number) => void;
}

export function DialItem({
  persona,
  index,
  isSelected,
  entranceDelay,
  rotation,
  onPick,
}: Props) {
  const reduceMotion = useReducedMotion();
  const angle = index * DIAL_STEP;
  // Derived from the rotation MotionValue so drags and springs move items
  // without React re-renders.
  const transform = useTransform(
    rotation,
    (r) =>
      `translate(-50%, -50%) rotate(${angle}deg) translateY(${DIAL_RADIUS}px) rotate(${-angle - r}deg)`,
  );
  const fade = useTransform(
    rotation,
    (r) => 1 - Math.min(stepsFromBottom(index, r), 3) * 0.12,
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div
          data-dial-index={index}
          className="absolute left-1/2 top-1/2 size-32"
          style={{ transform }}
        >
          <motion.span
            className="block size-full"
            initial={{
              opacity: 0,
              filter: reduceMotion ? "blur(0px)" : "blur(10px)",
            }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{
              duration: 0.35,
              ease: [0.16, 1, 0.3, 1],
              delay: reduceMotion ? 0 : entranceDelay,
            }}
          >
            <motion.button
              type="button"
              role="option"
              aria-selected={isSelected}
              aria-label={`${persona.name} — ${persona.role}`}
              onClick={() => onPick(index)}
              className="flex size-full items-center justify-center overflow-hidden rounded-full border"
              style={{
                borderColor: persona.accent,
                backgroundColor: persona.tint,
                opacity: fade,
              }}
            >
              {persona.image ? (
                <Image
                  src={persona.image}
                  alt=""
                  width={128}
                  height={128}
                  className="pointer-events-none size-full rounded-full object-cover"
                />
              ) : (
                <AutoGPTLogo
                  hideText
                  viewBox="47 -1 42 42"
                  className="pointer-events-none size-16"
                />
              )}
            </motion.button>
          </motion.span>
        </motion.div>
      </TooltipTrigger>
      <TooltipContent side="top">
        {persona.name} — {persona.role}
      </TooltipContent>
    </Tooltip>
  );
}
