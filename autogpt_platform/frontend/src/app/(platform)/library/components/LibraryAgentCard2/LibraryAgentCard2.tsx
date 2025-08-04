"use client";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/ui/separator";
import { CaretCircleRightIcon, CircleNotchIcon } from "@phosphor-icons/react";
import Image from "next/image";

interface LibraryAgentCard2Props {
  id: string;
  title: string;
  imageUrl: string;
  lastRunTime: string;
  totalRuns: number;
  runningAgents: number;
  source: string;
}

export const LibraryAgentCard2 = ({
  id,
  title,
  imageUrl,
  lastRunTime,
  totalRuns,
  runningAgents,
  source = "Marketplace",
}: LibraryAgentCard2Props) => {
  return (
    <div className="h-44 space-y-2 rounded-medium bg-white p-2 pl-3">
      {/* Destination */}
      <div className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-full bg-green-400" />
        <Text
          variant="small-medium"
          className="uppercase !leading-5 tracking-[0.1em] !text-zinc-400"
        >
          {source}
        </Text>
      </div>

      {/* Information */}
      <div className="flex gap-4 pb-1">
        <div className="flex flex-1 flex-col justify-between">
          <Text variant="large-medium" className="line-clamp-2">
            {title}
          </Text>
          <div className="flex flex-row justify-between gap-2">
            <Text variant="small" className="flex-1 !leading-5 !text-zinc-400">
              {lastRunTime}
            </Text>
            <Text variant="small" className="flex-1 !leading-5 !text-zinc-400">
              {totalRuns} runs
            </Text>
          </div>
        </div>
        <div className="relative aspect-video h-[4.75rem] overflow-hidden rounded-small">
          <Image
            src={imageUrl}
            alt="Agent-image"
            fill
            className="object-cover"
          />
        </div>
      </div>

      <Separator className="border-zinc-200" />

      {/* Actions */}
      <div className="flex gap-2">
        <Link
          href={`/library/agent/${id}`}
          className="flex w-1/3 flex-row gap-2 !text-xs"
        >
          <span className="group inline-flex items-center gap-1 text-neutral-800">
            See runs
            <CaretCircleRightIcon
              size={20}
              className="transition-transform duration-200 group-hover:translate-x-1"
            />
          </span>
        </Link>
        <Link
          href={`/library/agent/${id}`}
          className="flex w-1/3 flex-row gap-2 !text-xs"
        >
          <span className="group inline-flex items-center gap-1 text-neutral-800">
            Open in builder
            <CaretCircleRightIcon
              size={20}
              className="transition-transform duration-200 group-hover:translate-x-1"
            />
          </span>
        </Link>
        {runningAgents > 0 && (
          <Text
            variant="small"
            className="flex w-1/3 items-center justify-end gap-1 !leading-5 !text-zinc-400"
          >
            {runningAgents} agent{runningAgents !== 1 ? "s" : ""} running
            <CircleNotchIcon size={20} className="animate-spin" />
          </Text>
        )}
      </div>
    </div>
  );
};
