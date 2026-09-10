import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
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
import {
  AUTOPILOT_AVATAR_BG_CLASS,
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_BLURB,
  AUTOPILOT_NAME,
  AUTOPILOT_PILL_CLASS,
  AUTOPILOT_ROLE,
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
      aria-label={AUTOPILOT_NAME}
      className="flex flex-col overflow-hidden rounded-2xl bg-white smooth-shadow-ring-sm"
    >
      <Link
        href="/team/autopilot"
        aria-label={`View ${AUTOPILOT_NAME}`}
        className="flex flex-1 flex-col items-start p-2 pb-4"
      >
        <ExpertCover color={undefined} status="built-in" />

        <div className="flex w-full items-start gap-3 px-2">
          <Avatar className="relative z-10 -mt-12 ml-1 size-[5.5rem] shrink-0 rounded-full border border-black ring-4 ring-white">
            <AvatarImage
              src={AUTOPILOT_AVATAR_URL}
              alt={AUTOPILOT_NAME}
              width={88}
              height={88}
              className={AUTOPILOT_AVATAR_BG_CLASS}
            />
            <AvatarFallback className="grain-overlay">
              {AUTOPILOT_NAME}
            </AvatarFallback>
          </Avatar>
        </div>

        <div className="mt-2 flex w-full flex-col items-start gap-1 px-2 pl-5 text-left">
          <Text
            variant="lead-medium"
            tone="primary"
            className="w-full truncate pb-1"
          >
            {AUTOPILOT_NAME}
          </Text>
          <Text
            variant="small-medium"
            as="span"
            className={cn(
              "inline-flex items-center gap-1.5 self-start rounded-full px-2.5 py-0.5",
              AUTOPILOT_PILL_CLASS,
            )}
          >
            <Icon icon={SparklesIcon} size={12} />
            {AUTOPILOT_ROLE}
          </Text>
          <Text variant="body" tone="muted" className="mt-1 line-clamp-3">
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
