import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { creditsToUsdLabel } from "@/lib/credits";
import Link from "next/link";
import {
  OUTLINE_ACTION_BUTTON_CLASS,
  getNeedsSetupCount,
  getScheduleCountLabel,
  getWeeklySpend,
} from "../../helpers";

interface Props {
  experts: Expert[];
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
}

const CELL_CLASS = "px-4 py-3 align-middle";

export function ExpertTable({ experts, schedulesForExpert }: Props) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-zinc-200 bg-white">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-zinc-100 bg-zinc-50/60">
            <th scope="col" className={`${CELL_CLASS} font-medium`}>
              <Text variant="small-medium" className="text-zinc-600">
                Expert
              </Text>
            </th>
            <th scope="col" className={`${CELL_CLASS} font-medium`}>
              <Text variant="small-medium" className="text-zinc-600">
                Workflows
              </Text>
            </th>
            <th scope="col" className={`${CELL_CLASS} font-medium`}>
              <Text variant="small-medium" className="text-zinc-600">
                Schedules
              </Text>
            </th>
            <th scope="col" className={`${CELL_CLASS} font-medium`}>
              <Text variant="small-medium" className="text-zinc-600">
                Spend this week
              </Text>
            </th>
            <th scope="col" className={CELL_CLASS}>
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {experts.map((expert) => (
            <ExpertRow
              key={expert.id}
              expert={expert}
              schedules={schedulesForExpert(expert)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface RowProps {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
}

function ExpertRow({ expert, schedules }: RowProps) {
  const workflowCount = expert.workflows.length;
  const needsSetupCount = getNeedsSetupCount(expert, schedules);
  const weeklySpend = getWeeklySpend(expert);

  return (
    <tr className="border-b border-zinc-100 last:border-b-0 hover:bg-zinc-50/60">
      <td className={CELL_CLASS}>
        <Link
          href={`/team/${expert.id}`}
          aria-label={`View ${expert.name}`}
          className="flex items-center gap-3"
        >
          <Avatar className="h-9 w-9 shrink-0 rounded-lg">
            {expert.avatar_url ? (
              <AvatarImage
                src={expert.avatar_url}
                alt={expert.name}
                width={36}
                height={36}
                className="bg-white"
              />
            ) : null}
            <AvatarFallback square className="grain-overlay">
              {expert.name}
            </AvatarFallback>
          </Avatar>
          <div className="min-w-0">
            <Text variant="body-medium">{expert.name}</Text>
            <Text variant="small" className="text-zinc-500">
              {expert.role}
            </Text>
          </div>
        </Link>
      </td>
      <td className={CELL_CLASS}>
        <div className="flex items-center gap-2">
          <Text variant="small" className="text-zinc-600">
            {workflowCount}
          </Text>
          {needsSetupCount > 0 ? (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 ring-1 ring-inset ring-amber-200">
              {needsSetupCount} {needsSetupCount === 1 ? "needs" : "need"} setup
            </span>
          ) : null}
        </div>
      </td>
      <td className={CELL_CLASS}>
        <Text variant="small" className="text-zinc-600">
          {getScheduleCountLabel(schedules) ?? "No schedules yet"}
        </Text>
      </td>
      <td className={CELL_CLASS}>
        <Text
          variant="small"
          unmask={false}
          className="tabular-nums text-zinc-600"
        >
          {weeklySpend
            ? `${creditsToUsdLabel(weeklySpend.spent)} / ${creditsToUsdLabel(weeklySpend.budget)}`
            : "No budget"}
        </Text>
      </td>
      <td className={`${CELL_CLASS} text-right`}>
        <Button
          as="NextLink"
          href={`/copilot?expertId=${expert.id}`}
          variant="outline"
          size="small"
          className={OUTLINE_ACTION_BUTTON_CLASS}
        >
          Chat
        </Button>
      </td>
    </tr>
  );
}
