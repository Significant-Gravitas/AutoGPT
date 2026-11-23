"use client";

import { EvalsScreen } from "../affect/evals";
import {
  modelLabel,
  runMeta,
  toChanges,
  toRunOptions,
  toScreenCases,
  toStats,
  toVersions,
} from "../helpers";
import { useEvalsPage } from "../useEvalsPage";
import { CaseSidePanel } from "./CaseSidePanel/CaseSidePanel";

/** Feeds the saved eval runs into the Affect UI Evals block, with the open
 *  cases in a side panel like the chat's work area. */
export function EvalsContent() {
  const page = useEvalsPage();
  const detail = page.detail;
  const current = page.runs.find((run) => run.id === page.runId);
  const withCases = page.runs.find(
    (run) => run.case_coverage !== "none" && run.id !== page.runId,
  );

  return (
    <div className="flex h-full w-full overflow-hidden">
      <div className="h-full min-w-0 flex-1">
        <EvalsScreen
          title={current?.label ?? "Evals"}
          model={current ? modelLabel(current) : ""}
          meta={current ? runMeta(current) : ""}
          runId={page.runId}
          runs={toRunOptions(page.runs)}
          onPickRun={page.pickRun}
          commands={detail?.commands ?? page.commands}
          versions={detail ? toVersions(detail) : []}
          stats={detail ? toStats(detail) : []}
          cases={detail ? toScreenCases(detail.cases) : []}
          changes={detail ? toChanges(detail) : null}
          note={detail?.note}
          selected={page.caseTabs.isOpen ? page.caseTabs.active : null}
          onSelect={page.caseTabs.open}
          loading={page.isLoadingRuns || page.isLoadingRun}
          empty={!page.isLoadingRuns && !page.runsError && !page.runs.length}
          error={errorOf(page)}
          emptyCases={{
            text: "Only this run's totals and category scores were kept.",
            action: withCases
              ? {
                  label: `Open ${withCases.label}`,
                  onClick: () => page.pickRun(withCases.id),
                }
              : undefined,
          }}
        />
      </div>
      <CaseSidePanel tabs={page.caseTabs} casesById={page.casesById} />
    </div>
  );
}

function errorOf(page: ReturnType<typeof useEvalsPage>) {
  if (page.runsError) {
    return {
      title: "We couldn't load the saved runs",
      text: "The eval results could not be listed. Try again in a moment.",
      onRetry: () => void page.retryRuns(),
    };
  }
  if (page.runError) {
    return {
      title: "We couldn't load this run",
      text: "Its result files may have moved, or may not be valid JSON.",
      onRetry: () => void page.retryRun(),
    };
  }
  return null;
}
