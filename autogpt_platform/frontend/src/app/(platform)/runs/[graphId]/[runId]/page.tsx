"use client";

import { useParams } from "next/navigation";

import { RunSetup } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/components/RunSetup";

import { RunNotFound } from "../../components/RunNotFound";
import { RunScreen } from "../../components/RunScreen";
import { StepSidePanel } from "../../components/StepSidePanel";
import { useRunPage } from "../../useRunPage";

/** One workflow run on a page of its own, on the Affect UI Run Detail
 *  block: the trace across the width, an open block run beside it. */
export default function RunPage() {
  const params = useParams<{ graphId: string; runId: string }>();
  const page = useRunPage(params.graphId, params.runId);

  if (page.notFound) {
    return <RunNotFound error={page.error} onRetry={page.retry} />;
  }

  return (
    <>
      <div className="h-(--inset-height) min-h-0 w-full overflow-hidden">
        {page.isSettingUp && page.agent ? (
          <div className="mx-auto w-full max-w-xl p-6">
            <RunSetup
              agent={page.agent}
              onCancel={page.cancelSetup}
              onStarted={page.openRun}
            />
          </div>
        ) : (
          <RunScreen
            {...page.data}
            opened={page.tabs.isOpen ? page.tabs.active : null}
            onOpen={page.tabs.open}
            onRerun={page.controls.rerun}
            onStop={page.controls.stop}
          />
        )}
      </div>
      <StepSidePanel tabs={page.tabs} data={page.data} />
    </>
  );
}
