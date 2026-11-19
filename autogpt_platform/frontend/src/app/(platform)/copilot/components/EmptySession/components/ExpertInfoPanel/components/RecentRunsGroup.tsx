import { BlockRow } from "../../../../WorkArea/blocks";
import { StatusMark } from "../../../../WorkArea/WorkflowRunTab/components/StatusMark";
import {
  formatWhen,
  statusView,
  triggerLabel,
} from "../../../../WorkArea/WorkflowRunTab/runs";
import { RUN_TONE_STATE } from "../helpers";
import { useRecentRuns } from "../useRecentRuns";
import {
  InfoEmpty,
  InfoGroup,
  InfoList,
  InfoLoading,
  InfoMeta,
} from "./InfoParts";

interface Props {
  /** Only this expert's runs; null for all of the user's own. */
  expertID: string | null;
}

/** The newest workflow runs: which workflow, how it started, when and how it
 *  ended. A row opens the run beside the chat. */
export function RecentRunsGroup({ expertID }: Props) {
  const { runs, isLoading, openRun } = useRecentRuns(expertID);

  return (
    <InfoGroup label="Recent runs">
      {isLoading ? (
        <InfoLoading />
      ) : runs.length === 0 ? (
        <InfoEmpty>No runs yet.</InfoEmpty>
      ) : (
        <InfoList>
          {runs.map(({ run, name }) => {
            const view = statusView(run.status);
            return (
              <BlockRow
                key={run.id}
                tile={<StatusMark state={RUN_TONE_STATE[view.tone]} />}
                title={name}
                subtitle={`${triggerLabel(run)} · ${formatWhen(run.started_at)}`}
                trailing={<InfoMeta>{view.label}</InfoMeta>}
                onClick={() => openRun(run, name)}
              />
            );
          })}
        </InfoList>
      )}
    </InfoGroup>
  );
}
