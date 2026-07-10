import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowRightIcon,
  CheckCircleIcon,
  ClockIcon,
} from "@phosphor-icons/react";
import { Fragment } from "react";
import type { TourAgent } from "../../script/types";

interface Props {
  agent: TourAgent;
  /** Flips the badge from an animated "Running" to "Run completed". */
  runCompleted: boolean;
}

export function TourAgentCard({ agent, runCompleted }: Props) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-zinc-200/70 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <Text variant="large-medium" className="text-zinc-900">
          {agent.name}
        </Text>
        <span className="flex shrink-0 items-center gap-2 rounded-full bg-emerald-100 px-3 py-1.5 text-sm font-semibold text-emerald-700">
          {runCompleted ? (
            <>
              <CheckCircleIcon className="size-4 shrink-0" weight="fill" />
              Run completed
            </>
          ) : (
            <>
              <span className="relative flex size-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-75" />
                <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
              </span>
              Running
            </>
          )}
        </span>
      </div>

      <div className="flex items-center gap-1.5 text-sm text-zinc-600">
        <ClockIcon className="size-4 shrink-0" />
        <span>{agent.schedule} · created from your sentence</span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {agent.blocks.map((block, index) => (
          <Fragment key={`${block}-${index}`}>
            {index > 0 && (
              <ArrowRightIcon className="size-3.5 shrink-0 text-zinc-400" />
            )}
            <span className="rounded-md bg-zinc-100 px-2.5 py-1 font-mono text-xs text-zinc-700">
              {block}
            </span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}
