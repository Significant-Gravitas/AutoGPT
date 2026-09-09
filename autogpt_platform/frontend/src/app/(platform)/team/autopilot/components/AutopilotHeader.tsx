import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { cn } from "@/lib/utils";
import { AUTOPILOT_PILL_CLASS, AUTOPILOT_ROLE } from "../../helpers";

export function AutopilotHeader() {
  return (
    <header>
      <ExpertCover className="h-36" color={undefined} status="built-in" />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <span className="relative z-10 -mt-12 ml-14 flex size-24 shrink-0 items-center justify-center rounded-full bg-white ring-4 ring-white">
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-12" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
              Autopilot
            </h1>
            <Text
              variant="body-medium"
              as="span"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-0.5",
                AUTOPILOT_PILL_CLASS,
              )}
            >
              <Icon icon={SparklesIcon} size={14} />
              {AUTOPILOT_ROLE}
            </Text>
          </div>
        </div>
        <Button
          as="NextLink"
          href="/copilot"
          variant="primary"
          size="xs"
          className="shrink-0"
        >
          Chat
        </Button>
      </div>
    </header>
  );
}
