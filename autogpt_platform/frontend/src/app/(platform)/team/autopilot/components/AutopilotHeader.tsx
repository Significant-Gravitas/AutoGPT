import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Tick02Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { cn } from "@/lib/utils";
import { AUTOPILOT_PILL_CLASS, AUTOPILOT_ROLE } from "../../helpers";
import {
  AUTOPILOT_COVER_COLOR,
  AUTOPILOT_COVER_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";

export function AutopilotHeader() {
  return (
    <header>
      <ExpertCover
        className="h-36"
        color={AUTOPILOT_COVER_COLOR}
        art={AUTOPILOT_COVER_URL}
      />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <AutopilotAvatar
          size={96}
          className="relative z-10 -mt-12 ml-14 ring-4 ring-white"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
              {AUTOPILOT_NAME}
            </h1>
            <Text
              variant="small-medium"
              as="span"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5",
                AUTOPILOT_PILL_CLASS,
              )}
            >
              <Icon icon={SparklesIcon} size={12} />
              {AUTOPILOT_ROLE}
            </Text>
            <Text
              variant="small-medium"
              as="span"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5",
                AUTOPILOT_PILL_CLASS,
              )}
            >
              <Icon icon={Tick02Icon} size={12} />
              Built in
            </Text>
          </div>
        </div>
        <Button
          as="NextLink"
          href="/copilot"
          variant="primary"
          size="small"
          className="shrink-0"
        >
          Chat
        </Button>
      </div>
    </header>
  );
}
