import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import {
  AUTOPILOT_BLURB,
  AUTOPILOT_ROLE,
  ACTION_BUTTON_CLASS,
} from "../helpers";
import { CardStat, CardStats } from "./CardStats";
import { ExpertCover } from "./ExpertTeamCard/components/ExpertCover";

interface Props {
  skillCount: number;
  scheduleCount: number;
  workflowCount: number;
  onChat: () => void;
}

export function AutopilotCard({
  skillCount,
  scheduleCount,
  workflowCount,
  onChat,
}: Props) {
  return (
    <section
      aria-label="Autopilot"
      className="flex flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white"
    >
      <Link
        href="/team/autopilot"
        aria-label="View Autopilot"
        className="flex flex-1 flex-col items-start p-2 pb-4"
      >
        <ExpertCover
          color={undefined}
          src="/experts/covers/autopilot.jpg"
          status="built-in"
        />

        <div className="flex w-full items-start gap-3 px-2">
          <span className="relative z-10 -mt-11 ml-1 flex size-[5.25rem] shrink-0 items-center justify-center rounded-full bg-white ring-4 ring-white">
            <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-11" />
          </span>
        </div>

        <div className="mt-2 flex w-full flex-col items-start gap-1 px-2 pl-5 text-left">
          <Text variant="h4" className="w-full truncate pb-1">
            Autopilot
          </Text>
          <Text variant="body" className="text-zinc-500">
            {AUTOPILOT_ROLE}
          </Text>
          <Text variant="body" className="mt-1 line-clamp-3 text-zinc-500">
            {AUTOPILOT_BLURB}
          </Text>
        </div>

        <div className="w-full px-2">
          <CardStats className="mt-3 w-full">
            <CardStat label="Schedules">{scheduleCount}</CardStat>
            <CardStat label="Skills">{skillCount}</CardStat>
            <CardStat label="Workflows">{workflowCount}</CardStat>
          </CardStats>
        </div>
      </Link>

      <div className="flex items-center gap-2 px-4 pb-4">
        <Button
          variant="secondary"
          size="small"
          className={`${ACTION_BUTTON_CLASS} flex-1`}
          leftIcon={<Icon icon={PencilEdit02Icon} size={14} />}
          onClick={onChat}
        >
          Chat
        </Button>
      </div>
    </section>
  );
}
