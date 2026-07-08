"use client";

import { cn } from "@/lib/utils";
import {
  ArrowsClockwiseIcon,
  CaretDownIcon,
  CheckCircleIcon,
  StrategyIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useId, useState } from "react";
import { OrbitLoader } from "../OrbitLoader/OrbitLoader";
import { ExecutorPromptCollapse } from "./ExecutorPromptCollapse";
import { PlanSteps } from "./PlanSteps";
import {
  getPhaseLabel,
  isPlanningInFlight,
  shortModelName,
  type PlanPartData,
} from "./helpers";

interface Props {
  data: PlanPartData;
}

function PhaseIcon({ data }: Props) {
  if (isPlanningInFlight(data)) return <OrbitLoader size={16} />;
  switch (data.phase) {
    case "replanned":
    case "replan_capped":
    case "replan_failed":
      return <ArrowsClockwiseIcon className="size-4 text-amber-600" />;
    case "failed":
    case "skipped":
      return <WarningCircleIcon className="size-4 text-zinc-400" />;
    default:
      return <StrategyIcon className="size-4 text-violet-600" />;
  }
}

/**
 * Live plan card for the two-phase planner/executor split. Renders the
 * `data-plan` UI-message part through its lifecycle: a spinner while the
 * planner runs, then the decided steps (with per-step success criteria and
 * expected tools), the exact prompt handed to the executor, and any
 * mid-turn re-plan notices.
 */
export function PlanCard({ data }: Props) {
  const contentId = useId();
  const shouldReduceMotion = useReducedMotion();
  const hasDetail = data.steps.length > 0 || data.executorPrompt != null;
  const [isExpanded, setIsExpanded] = useState(true);

  const planner = shortModelName(data.plannerModel);
  const executor = shortModelName(data.executorModel);
  const isNotice = data.phase === "skipped" || data.phase === "failed";

  return (
    <div
      className={cn(
        "mt-2 w-full rounded-xl border px-3 py-2.5",
        isNotice
          ? "border-zinc-200 bg-zinc-50"
          : "border-violet-100 bg-violet-50/40",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-2.5">
          <span className="mt-0.5 flex shrink-0 items-center">
            <PhaseIcon data={data} />
          </span>
          <div className="min-w-0">
            <p className="text-sm font-medium text-zinc-800">
              {getPhaseLabel(data)}
            </p>
            {(planner || executor) && (
              <p className="mt-0.5 truncate text-xs text-zinc-500">
                {planner && <span>Planner: {planner}</span>}
                {planner && executor && <span> · </span>}
                {executor && <span>Executor: {executor}</span>}
              </p>
            )}
            {data.reason && (
              <p className="mt-0.5 text-xs text-zinc-500">{data.reason}</p>
            )}
          </div>
        </div>

        {hasDetail && (
          <button
            type="button"
            aria-expanded={isExpanded}
            aria-controls={contentId}
            onClick={() => setIsExpanded((v) => !v)}
            className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-zinc-500 hover:bg-white/70"
          >
            {isExpanded ? "Hide" : "Show plan"}
            <CaretDownIcon
              className={cn(
                "size-3.5 transition-transform",
                isExpanded && "rotate-180",
              )}
              weight="bold"
            />
          </button>
        )}
      </div>

      <AnimatePresence initial={false}>
        {hasDetail && isExpanded && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : { duration: 0.35, ease: [0.16, 1, 0.3, 1] }
            }
            className="overflow-hidden"
          >
            <div className="pt-3">
              <PlanSteps steps={data.steps} />
              {data.phase === "replan_capped" && (
                <p className="mt-2 flex items-center gap-1.5 text-xs text-amber-700">
                  <CheckCircleIcon className="size-3.5" />
                  Continuing best-effort without further re-planning.
                </p>
              )}
              {data.executorPrompt && (
                <ExecutorPromptCollapse prompt={data.executorPrompt} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
