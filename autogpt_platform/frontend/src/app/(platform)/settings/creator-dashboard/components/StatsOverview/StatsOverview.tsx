"use client";
import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT, formatRuns, type DashboardStats } from "../../helpers";
import {
  CheckmarkCircle02Icon,
  Clock01Icon,
  Layers01Icon,
  PlayIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";

interface Props {
  stats: DashboardStats;
  index?: number;
}

interface StatTile {
  label: string;
  value: string;
  Icon: IconSvgElement;
  iconClass: string;
}

export function StatsOverview({ stats, index = 0 }: Props) {
  const reduceMotion = useReducedMotion();

  const tiles: StatTile[] = [
    {
      label: "Total submissions",
      value: stats.total.toLocaleString(),
      Icon: Layers01Icon,
      iconClass: "bg-accent-surface text-accent-strong ring-brand/20",
    },
    {
      label: "Approved",
      value: stats.approved.toLocaleString(),
      Icon: CheckmarkCircle02Icon,
      iconClass: "bg-success-surface text-success-foreground ring-success/30",
    },
    {
      label: "In review",
      value: stats.pending.toLocaleString(),
      Icon: Clock01Icon,
      iconClass:
        "bg-warning-surface text-warning-foreground ring-warning-border",
    },
    {
      label: "Total runs",
      value: formatRuns(stats.totalRuns),
      Icon: PlayIcon,
      // oxlint-disable-next-line design-system/no-raw-colors -- sky tile kept until the cyan -> info hue change is approved
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
        <div className="col-span-2 flex items-center justify-between gap-3 rounded-large border border-border-base bg-gradient-to-br from-warning-surface/60 to-surface px-4 py-4 shadow-message-table lg:col-span-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-warning/20 text-warning-foreground ring-1 ring-inset ring-warning-border">
              <UIIcon icon={StarIcon} size={18} />
            </div>
            <div className="flex flex-col">
              <Text variant="body-medium" as="span">
                Average rating
              </Text>
              <Text variant="small" tone="muted">
                Across all submissions with reviews.
              </Text>
            </div>
          </div>
          <Text variant="h3" as="span" size="large-medium">
            {stats.averageRating.toFixed(1)}
            <span className="text-foreground-subtle"> / 5</span>
          </Text>
        </div>
      ) : null}
    </motion.section>
  );
}

function StatTileCard({ tile }: { tile: StatTile }) {
  const { label, value, iconClass } = tile;
  return (
    <div className="relative flex min-h-[120px] flex-col justify-between gap-2 rounded-large border border-border-base bg-surface px-4 py-5 shadow-message-table">
      <div
        className={`absolute right-3 top-3 flex h-7 w-7 items-center justify-center rounded-full ring-1 ring-inset ${iconClass}`}
      >
        <UIIcon icon={tile.Icon} size={14} />
      </div>
      <Text variant="body" as="span" tone="emphasis" className="mr-10">
        {label}
      </Text>
      <Text variant="h4" as="span">
        {value}
      </Text>
    </div>
  );
}
