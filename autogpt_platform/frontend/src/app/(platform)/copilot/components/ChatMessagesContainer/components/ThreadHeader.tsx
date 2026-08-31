import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import type { ExpertIdentity } from "../../../useExpertMap";
import { ExpertAvatar } from "./ExpertAvatar/ExpertAvatar";
import { ExpertSchedulesButton } from "./ExpertSchedulesButton/ExpertSchedulesButton";

// Autopilot's product-facing title when a session carries no expert identity.
const DEFAULT_EXPERT_ROLE = "Head of AI";

interface Props {
  expertIdentity?: ExpertIdentity | null;
  readOnly: boolean;
  /** The workspace-files card floats over the column's right side, so the
   *  chip slides left to make room for it. */
  areFilesOpen: boolean;
  /** The new layout floats the sidebar and workspace-files controls over the
   *  chat's top-left corner below `lg`; the chip clears them. */
  hasFloatingControls?: boolean;
}

/** Floats as a translucent chip over the scroller's top edge — the
 *  zero-height wrapper keeps it out of the flex flow so messages scroll
 *  underneath. An expert session wears the expert's identity and every
 *  other session is Autopilot's, so the thread is never anonymous. */
export function ThreadHeader({
  expertIdentity,
  readOnly,
  areFilesOpen,
  hasFloatingControls = false,
}: Props) {
  const name = expertIdentity?.name ?? "Autopilot";
  const role = expertIdentity?.role ?? DEFAULT_EXPERT_ROLE;

  return (
    <div data-testid="expert-thread-header" className="relative z-20 h-0">
      <div
        className={cn(
          "ease-[cubic-bezier(0.32,0.72,0,1)] pointer-events-none absolute inset-x-0 top-3 mx-auto flex w-full max-w-3xl justify-start px-6 transition-transform duration-300 will-change-transform motion-reduce:transition-none",
          areFilesOpen && "xl:-translate-x-40",
          hasFloatingControls && "max-md:pl-28 md:max-lg:pl-20",
        )}
      >
        <div className="pointer-events-auto flex min-w-0 items-center gap-1.5 rounded-full border border-zinc-200/70 bg-white/75 py-1 pl-1.5 pr-3 shadow-sm backdrop-blur-md">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex min-w-0 items-center gap-2">
                  <ExpertAvatar
                    name={name}
                    avatarUrl={expertIdentity?.avatarUrl ?? null}
                    isAutopilot={!expertIdentity}
                    size="sm"
                  />
                  <span className="truncate text-sm font-medium text-zinc-800">
                    {name}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">{role}</TooltipContent>
            </Tooltip>
            {expertIdentity && !readOnly && !expertIdentity.isArchived && (
              <ExpertSchedulesButton
                expertId={expertIdentity.id}
                expertName={expertIdentity.name}
              />
            )}
          </TooltipProvider>
        </div>
      </div>
    </div>
  );
}
