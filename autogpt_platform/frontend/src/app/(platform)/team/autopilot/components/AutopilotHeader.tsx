import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Tick02Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { AUTOPILOT_ROLE } from "../../helpers";
import {
  AUTOPILOT_COVER_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import { AUTOPILOT_HEX } from "@/components/molecules/ExpertAvatar/colors";

export function AutopilotHeader() {
  return (
    <header>
      <ExpertCover
        className="h-36"
        color={AUTOPILOT_HEX}
        art={AUTOPILOT_COVER_URL}
      />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <AutopilotAvatar
          size={96}
          transparent={false}
          // oxlint-disable-next-line design-system/no-restyle -- AutopilotAvatar has no bordered variant (src/components is P9's)
          className="relative z-10 -mt-12 ml-4 rounded-full border border-border-base"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-heading text-foreground-strong">
              {AUTOPILOT_NAME}
            </h1>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border-base bg-surface-muted px-2.5 py-0.5 font-sans text-caption font-medium text-foreground-body">
              <Icon icon={SparklesIcon} size={12} />
              {AUTOPILOT_ROLE}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border-base bg-surface-muted px-2.5 py-0.5 font-sans text-caption font-medium text-foreground-body">
              <Icon icon={Tick02Icon} size={12} />
              Built in
            </span>
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
