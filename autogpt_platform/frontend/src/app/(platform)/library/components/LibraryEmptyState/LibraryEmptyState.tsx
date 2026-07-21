"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PlusIcon, StorefrontIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

const EASE_OUT_QUINT = [0.22, 1, 0.36, 1] as const;

export function LibraryEmptyState() {
  const shouldReduceMotion = useReducedMotion();

  function fadeUp(delay: number) {
    if (shouldReduceMotion) {
      return {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        transition: { duration: 0.2, delay: 0 },
      };
    }
    return {
      initial: { opacity: 0, y: 8 },
      animate: { opacity: 1, y: 0 },
      transition: { duration: 0.35, ease: EASE_OUT_QUINT, delay },
    };
  }

  return (
    <div className="flex flex-col items-center justify-center gap-8 px-6 py-16 text-center">
      <motion.div {...fadeUp(0)}>
        <StackedCardsIllustration />
      </motion.div>

      <div className="flex max-w-md flex-col items-center gap-2">
        <motion.div {...fadeUp(0.28)}>
          <Text variant="h3" className="text-zinc-900">
            Your library is empty
          </Text>
        </motion.div>
        <motion.div {...fadeUp(0.36)}>
          <Text variant="body" className="text-zinc-500">
            Build your own agent from scratch, or grab one from the marketplace
            to get started.
          </Text>
        </motion.div>
      </div>

      <div className="flex flex-col items-center gap-3 sm:flex-row">
        <motion.div {...fadeUp(0.44)}>
          <Button
            as="NextLink"
            href="/build"
            variant="primary"
            size="large"
            leftIcon={<PlusIcon className="h-4 w-4" weight="bold" />}
          >
            Build an agent
          </Button>
        </motion.div>
        <motion.div {...fadeUp(0.5)}>
          <Button
            as="NextLink"
            href="/marketplace"
            variant="secondary"
            size="large"
            leftIcon={<StorefrontIcon className="h-4 w-4" weight="bold" />}
          >
            Browse marketplace
          </Button>
        </motion.div>
      </div>
    </div>
  );
}

const CARDS = [
  { x: 80, y: 8, width: 320, opacity: 0.55 },
  { x: 48, y: 48, width: 384, opacity: 0.8 },
  { x: 8, y: 92, width: 464, opacity: 1 },
];

function StackedCardsIllustration() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <svg
      width="480"
      height="170"
      viewBox="0 0 480 170"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className="select-none"
    >
      {CARDS.map((card, i) => (
        <motion.g
          key={i}
          initial={
            shouldReduceMotion
              ? { opacity: 0 }
              : { opacity: 0, y: 12, scale: 0.97 }
          }
          animate={
            shouldReduceMotion
              ? { opacity: card.opacity }
              : { opacity: card.opacity, y: 0, scale: 1 }
          }
          transition={{
            duration: shouldReduceMotion ? 0.2 : 0.45,
            ease: EASE_OUT_QUINT,
            delay: shouldReduceMotion ? 0 : i * 0.08,
          }}
          style={{ transformOrigin: "240px 96px", transformBox: "fill-box" }}
        >
          <Card x={card.x} y={card.y} width={card.width} />
        </motion.g>
      ))}
    </svg>
  );
}

type CardProps = {
  x: number;
  y: number;
  width: number;
};

function Card({ x, y, width }: CardProps) {
  const height = 56;
  const radius = 14;
  const padding = 16;
  const dotRadius = 8;

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={radius}
        ry={radius}
        fill="white"
        stroke="#E4E4E7"
        strokeWidth={1}
      />
      <circle
        cx={x + padding + dotRadius}
        cy={y + height / 2}
        r={dotRadius}
        fill="#E4E4E7"
      />
      <rect
        x={x + padding + dotRadius * 2 + 12}
        y={y + height / 2 - 6}
        width={width * 0.35}
        height={12}
        rx={6}
        ry={6}
        fill="#E4E4E7"
      />
      <rect
        x={x + width - padding - 80}
        y={y + height / 2 - 6}
        width={12}
        height={12}
        rx={3}
        ry={3}
        fill="#E4E4E7"
      />
      <rect
        x={x + width - padding - 60}
        y={y + height / 2 - 6}
        width={12}
        height={12}
        rx={3}
        ry={3}
        fill="#E4E4E7"
      />
      <rect
        x={x + width - padding - 40}
        y={y + height / 2 - 6}
        width={32}
        height={12}
        rx={6}
        ry={6}
        fill="#E4E4E7"
      />
    </g>
  );
}
