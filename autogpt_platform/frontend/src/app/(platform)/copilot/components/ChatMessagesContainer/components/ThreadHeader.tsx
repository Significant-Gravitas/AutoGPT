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
   *  centred row slides left to make room for it. */
  areFilesOpen: boolean;
  /** The new layout floats the sidebar and workspace-files controls over the
   *  chat's top-left corner below `lg`; the row starts past them. */
  hasFloatingControls?: boolean;
}

/** Sits above the scroller rather than sticky inside it, so the bar spans the
 *  full chat width while its row stays aligned with the max-w-3xl message
 *  column. An expert session wears the expert's identity and every other
 *  session is Autopilot's, so the thread is never anonymous. */
export function ThreadHeader({
  expertIdentity,
  readOnly,
  areFilesOpen,
  hasFloatingControls = false,
}: Props) {
  return (
    <div
      data-testid="expert-thread-header"
      className="z-10 w-full border-b border-b-[#80808017] bg-[#fafafa]/80 backdrop-blur-md"
    >
      <div
        className={cn(
          "ease-[cubic-bezier(0.32,0.72,0,1)] mx-auto flex w-full max-w-3xl items-center gap-2 px-6 py-2 transition-transform duration-300 will-change-transform motion-reduce:transition-none",
          areFilesOpen && "xl:-translate-x-40",
          hasFloatingControls && "max-md:pl-28 md:max-lg:pl-20",
        )}
      >
        <ExpertAvatar
          name={expertIdentity?.name ?? "Autopilot"}
          avatarUrl={expertIdentity?.avatarUrl ?? null}
          isAutopilot={!expertIdentity}
        />
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-sm font-medium text-zinc-800">
            {expertIdentity?.name ?? "Autopilot"}
          </span>
          <span className="truncate text-xs text-zinc-500">
            {expertIdentity?.role ?? DEFAULT_EXPERT_ROLE}
          </span>
        </div>
        {expertIdentity && !readOnly && !expertIdentity.isArchived && (
          <ExpertSchedulesButton
            expertId={expertIdentity.id}
            expertName={expertIdentity.name}
          />
        )}
      </div>
    </div>
  );
}
