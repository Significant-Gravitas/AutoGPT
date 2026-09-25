import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  BubbleChatIcon,
  Calendar03Icon,
  FlashIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { AUTOPILOT_BLURB, AUTOPILOT_ROLE } from "../helpers";
import {
  CHIP_SHAPE,
  CHIP_SIZE,
} from "@/app/(platform)/marketplace/components/CategoryChip/CategoryChip";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { CardStat, CardStats } from "./CardStats";
import { ExpertCover } from "./ExpertTeamCard/components/ExpertCover";

/** Otto's reserved lavender, the one colour no category can take. */
const AUTOPILOT_COVER_COLOR = "#B6A4C8";

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
      aria-label={AUTOPILOT_NAME}
      className="flex flex-col overflow-hidden rounded-2xl bg-white smooth-shadow-ring-sm"
    >
      <Link
        href="/team/autopilot"
        aria-label={`View ${AUTOPILOT_NAME}`}
        className="flex flex-1 flex-col items-start p-2 pb-4"
      >
        <ExpertCover color={AUTOPILOT_COVER_COLOR} status="built-in" />

        <div className="flex w-full items-start gap-3 px-2">
          <AutopilotAvatar
            size={88}
            backgroundColor={AUTOPILOT_COVER_COLOR}
            className="relative z-10 -mt-12 ml-1 rounded-full ring-4 ring-background"
          />
        </div>

        <div className="mt-2 flex w-full flex-col items-start gap-1 px-2 pl-5 text-left">
          <Text
            variant="lead-medium"
            tone="primary"
            className="w-full truncate pb-1"
          >
            {AUTOPILOT_NAME}
          </Text>
          {/* Not a category, but it sits in the same row as the experts'
              topic tags and should not read as a different kind of thing. */}
          <span
            className={cn(CHIP_SHAPE, CHIP_SIZE.small, "max-w-full")}
            style={{ color: AUTOPILOT_COVER_COLOR }}
          >
            <Icon icon={SparklesIcon} size={12} className="shrink-0" />
            <span className="truncate">{AUTOPILOT_ROLE}</span>
          </span>
          <Text
            variant="body"
            tone="muted"
            className="mt-1 line-clamp-2 min-h-[2lh]"
          >
            {AUTOPILOT_BLURB}
          </Text>
        </div>

        <div className="w-full px-2 pl-5">
          <CardStats className="mt-3 w-full">
            <CardStat
              icon={Calendar03Icon}
              label="Schedules"
              singular="schedule"
              count={scheduleCount}
            />
            <CardStat
              icon={SparklesIcon}
              label="Skills"
              singular="skill"
              count={skillCount}
            />
            <CardStat
              icon={FlashIcon}
              label="Workflows"
              singular="workflow"
              count={workflowCount}
            />
          </CardStats>
        </div>
      </Link>

      <div className="flex items-center gap-2 px-4 pb-4">
        <Button
          variant="secondary"
          size="small"
          className="flex-1"
          leadingIcon={BubbleChatIcon}
          onClick={onChat}
        >
          Chat
        </Button>
      </div>
    </section>
  );
}
