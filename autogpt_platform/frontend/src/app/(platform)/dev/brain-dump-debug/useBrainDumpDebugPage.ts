import {
  peekIntroPath,
  type IntroPath,
} from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useState } from "react";
import { useDebugFinalize } from "./useDebugFinalize";
import { useRecordingSnapshot } from "./useRecordingSnapshot";
import { useStatusTimeline } from "./useStatusTimeline";
import { buildWaterfall } from "./waterfall";

interface Args {
  enabled: boolean;
}

export function useBrainDumpDebugPage({ enabled }: Args) {
  const { snapshot, refresh } = useRecordingSnapshot();
  const timeline = useStatusTimeline({ enabled });
  const finalize = useDebugFinalize(snapshot);

  const [introPath, setIntroPath] = useState<IntroPath | null>(null);
  useEffect(() => {
    setIntroPath(peekIntroPath());
  }, []);

  return {
    snapshot,
    refreshSnapshot: refresh,
    timeline,
    finalize,
    introPath,
    stages: buildWaterfall({ snapshot, seenAt: timeline.seenAt, introPath }),
  };
}
