import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { AUTOPILOT_ROLE, ACTION_BUTTON_CLASS } from "../../helpers";

export function AutopilotHeader() {
  return (
    <header>
      <ExpertCover
        className="h-36"
        color={undefined}
        src="/experts/covers/autopilot.jpg"
        status="built-in"
      />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <span className="relative z-10 -mt-12 ml-14 flex size-24 shrink-0 items-center justify-center rounded-full bg-white ring-4 ring-white">
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-12" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
              Autopilot
            </h1>
            <span className="inline-flex items-center gap-1.5 rounded-md border border-zinc-200 bg-zinc-50 px-2.5 py-0.5 text-sm font-medium text-zinc-700">
              <Icon icon={SparklesIcon} size={14} />
              {AUTOPILOT_ROLE}
            </span>
          </div>
          <p className="mt-1 text-sm text-zinc-500">
            Built in, always on your team.
          </p>
        </div>
        <Button
          as="NextLink"
          href="/copilot"
          variant="primary"
          size="small"
          className={`${ACTION_BUTTON_CLASS} shrink-0`}
        >
          Chat
        </Button>
      </div>
    </header>
  );
}
