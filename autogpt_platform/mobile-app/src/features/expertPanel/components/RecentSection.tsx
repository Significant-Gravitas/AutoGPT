import {
  AlertCircleIcon,
  CircleIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { ActivityIndicator } from "react-native";

import { Icon } from "@/components/Icon";
import { relativeTime } from "@/features/team/helpers";
import type { ExpertRun } from "@/lib/api/expertActivity";
import { useColors } from "@/lib/theme";

import { runTone } from "../helpers";
import { ActivityRow } from "./ActivityRow";
import { ActivitySection } from "./ActivitySection";

interface Props {
  isLoading: boolean;
  runs: ExpertRun[];
}

export function RecentSection({ isLoading, runs }: Props) {
  const colors = useColors();

  function renderMark(run: ExpertRun) {
    const tone = runTone(run.status);
    if (tone === "done") return <Icon icon={Tick02Icon} size={20} color={colors.success} />;
    if (tone === "failed") {
      return <Icon icon={AlertCircleIcon} size={20} color={colors.error} />;
    }
    if (tone === "running") {
      return <ActivityIndicator size="small" color={colors.textTertiary} />;
    }
    return <Icon icon={CircleIcon} size={20} color={colors.textTertiary} />;
  }

  function when(run: ExpertRun) {
    const at = run.ended_at ?? run.started_at;
    return at ? relativeTime(at) : null;
  }

  return (
    <ActivitySection title="Recent" isLoading={isLoading}>
      {runs.map((run) => (
        <ActivityRow
          key={run.execution_id}
          leading={renderMark(run)}
          title={run.agent_name}
          detail={when(run)}
        />
      ))}
    </ActivitySection>
  );
}
