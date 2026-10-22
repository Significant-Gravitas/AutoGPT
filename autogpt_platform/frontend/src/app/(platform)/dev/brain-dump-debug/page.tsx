"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { notFound } from "next/navigation";
import { RecordingStatePanel } from "./components/RecordingStatePanel";
import { ServerStatusPanel } from "./components/ServerStatusPanel";
import { TimingPanel } from "./components/TimingPanel";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { UploadQueuePanel } from "./components/UploadQueuePanel";
import { isProductionEnvironment } from "./helpers";
import { useBrainDumpDebugPage } from "./useBrainDumpDebugPage";

const MAIN_CLASS = "container flex min-h-screen flex-col gap-6 pb-20 pt-16";

export default function BrainDumpDebugPage() {
  const isProduction = isProductionEnvironment();

  const isAvailable = !isProduction;
  const { snapshot, refreshSnapshot, timeline, finalize, introPath, stages } =
    useBrainDumpDebugPage({ enabled: isAvailable });

  if (isProduction) notFound();

  return (
    <main className={MAIN_CLASS}>
      <header className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <Text variant="h3">Brain dump debug</Text>
          <Badge variant="error">dev only</Badge>
          {introPath ? (
            <Badge variant="info">intro path {introPath}</Badge>
          ) : null}
        </div>
        <Text variant="body" className="max-w-prose text-zinc-600">
          Local recording state, the upload queue, the server pipeline and the
          processing budget for the onboarding brain dump. Not reachable in
          production.
        </Text>
      </header>

      <RecordingStatePanel snapshot={snapshot} onRefresh={refreshSnapshot} />
      <UploadQueuePanel snapshot={snapshot} />
      <ServerStatusPanel
        dump={timeline.dump}
        isLoading={timeline.isLoading}
        isError={timeline.isError}
        finalizeResponse={finalize.response}
        finalizeRoundTripMs={finalize.roundTripMs}
        finalizeError={finalize.errorMessage}
        canFinalize={finalize.canRun}
        isFinalizing={finalize.isRunning}
        onFinalize={finalize.run}
      />
      <TranscriptPanel finalizeResponse={finalize.response} />
      <TimingPanel stages={stages} finalizeRoundTripMs={finalize.roundTripMs} />
    </main>
  );
}
