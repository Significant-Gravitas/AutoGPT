"use client";

import { motion, useReducedMotion } from "framer-motion";

const EASE_OUT_QUINT = [0.22, 1, 0.36, 1] as const;

// A compact take on the library's stacked-cards illustration, sized for a
// panel rather than a page: three ghost rows settling into place.
const CARDS = [
  { x: 40, y: 4, width: 160, opacity: 0.5 },
  { x: 22, y: 28, width: 196, opacity: 0.75 },
  { x: 4, y: 52, width: 232, opacity: 1 },
];

export function HomeEmptyIllustration() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <svg
      width="240"
      height="96"
      viewBox="0 0 240 96"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className="select-none"
    >
      {CARDS.map((card, index) => (
        <motion.g
          key={index}
          initial={
            shouldReduceMotion
              ? { opacity: 0 }
              : { opacity: 0, y: 8, scale: 0.97 }
          }
          animate={
            shouldReduceMotion
              ? { opacity: card.opacity }
              : { opacity: card.opacity, y: 0, scale: 1 }
          }
          transition={{
            duration: shouldReduceMotion ? 0.2 : 0.45,
            ease: EASE_OUT_QUINT,
            delay: shouldReduceMotion ? 0 : index * 0.08,
          }}
          className="origin-[120px_52px] [transform-box:fill-box]"
        >
          <Card x={card.x} y={card.y} width={card.width} />
        </motion.g>
      ))}
    </svg>
  );
}

interface CardProps {
  x: number;
  y: number;
  width: number;
}

function Card({ x, y, width }: CardProps) {
  const height = 40;
  const radius = 10;
  const padding = 12;
  const dotRadius = 6;
  const midY = y + height / 2;

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={radius}
        ry={radius}
        strokeWidth={1}
        className="fill-white stroke-zinc-200"
      />
      <circle
        cx={x + padding + dotRadius}
        cy={midY}
        r={dotRadius}
        className="fill-zinc-200"
      />
      <rect
        x={x + padding + dotRadius * 2 + 8}
        y={midY - 4}
        width={width * 0.36}
        height={8}
        rx={4}
        ry={4}
        className="fill-zinc-200"
      />
      <rect
        x={x + width - padding - 52}
        y={midY - 4}
        width={8}
        height={8}
        rx={2}
        ry={2}
        className="fill-zinc-200"
      />
      <rect
        x={x + width - padding - 38}
        y={midY - 4}
        width={8}
        height={8}
        rx={2}
        ry={2}
        className="fill-zinc-200"
      />
      <rect
        x={x + width - padding - 24}
        y={midY - 4}
        width={24}
        height={8}
        rx={4}
        ry={4}
        className="fill-zinc-200"
      />
    </g>
  );
}
