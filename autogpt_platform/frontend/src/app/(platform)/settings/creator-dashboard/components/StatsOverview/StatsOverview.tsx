"use client";

import {
  CheckCircleIcon,
  ClockIcon,
  PlayIcon,
  StarIcon,
  StackIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT, formatRuns, type DashboardStats } from "../../helpers";

interface Props {
  stats: DashboardStats;
  index?: number;
}

interface StatTile {
  label: string;
  value: string;
  Icon: PhosphorIcon;
  iconClass: string;
}

export function StatsOverview({ stats, index = 0 }: Props) {
  const reduceMotion = useReducedMotion();

  const tiles: StatTile[] = [
    {
      label: "Total submissions",
      value: stats.total.toLocaleString(),
      Icon: StackIcon,
      iconClass: "bg-violet-50 text-violet-700 ring-violet-200",
    },
    {
      label: "Approved",
      value: stats.approved.toLocaleString(),
      Icon: CheckCircleIcon,
      iconClass: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    },
    {
      label: "In review",
      value: stats.pending.toLocaleString(),
      Icon: ClockIcon,
      iconClass: "bg-amber-50 text-amber-700 ring-amber-200",
    },
    {
      label: "Total runs",
      value: formatRuns(stats.totalRuns),
      Icon: PlayIcon,
      iconClass: "bg-sky-50 text-sky-700 ring-sky-200",
    },
  ];

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : {
              duration: 0.32,
              ease: EASE_OUT,
              delay: 0.04 + index * 0.05,
            }
      }
      className="grid w-full grid-cols-2 gap-3 lg:grid-cols-4"
      aria-label="Submission stats"
    >
      {tiles.map((tile) => (
        <StatTileCard key={tile.label} tile={tile} />
      ))}

      {stats.averageRating !== null ? (
        <div className="col-span-2 flex items-center justify-between gap-3 rounded-[18px] border border-zinc-200 bg-gradient-to-br from-amber-50/60 to-white px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)] lg:col-span-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-amber-100 text-amber-700 ring-1 ring-inset ring-amber-200">
              <StarIcon size={18} weight="fill" />
            </div>
            <div className="flex flex-col">
              <Text variant="body-medium" as="span" className="text-textBlack">
                Average rating
              </Text>
              <Text variant="small" className="text-zinc-500">
                Across all submissions with reviews.
              </Text>
            </div>
          </div>
          <Text
            variant="h3"
            as="span"
            size="large-medium"
            className="text-textBlack"
          >
            {stats.averageRating.toFixed(1)}
            <span className="text-zinc-400"> / 5</span>
          </Text>
        </div>
      ) : null}
    </motion.section>
  );
}

function StatTileCard({ tile }: { tile: StatTile }) {
  const { label, value, Icon, iconClass } = tile;
  return (
    <div className="relative flex min-h-[120px] flex-col justify-between gap-2 rounded-[18px] border border-zinc-200 bg-white px-4 py-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
      <div
        className={`absolute right-3 top-3 flex h-7 w-7 items-center justify-center rounded-full ring-1 ring-inset ${iconClass}`}
      >
        <Icon size={14} weight="bold" />
      </div>
      <Text variant="body" as="span" className="pr-10 text-zinc-800">
        {label}
      </Text>
      <Text variant="h4" as="span" className="text-textBlack">
        {value}
      </Text>
    </div>
  );
}
