"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  AppSidebarTrigger,
  useAppSidebarToggle,
} from "@/components/layout/AppSidebar/components/AppSidebarTrigger/AppSidebarTrigger";
import { cn } from "@/lib/utils";
import type { ExpertIdentity } from "../../../useExpertMap";
import { ExpertAvatar } from "./ExpertAvatar/ExpertAvatar";

// Otto's product-facing title when a session carries no expert identity.
const DEFAULT_EXPERT_ROLE = "Head of AI";

export type FloatingControls = "newLayout";

// Below `lg` the sidebar button moves into the chip itself (see
// PlatformChrome) and the files and integrations buttons onto the composer
// (ComposerPanelTab), so the chip stands alone, centred near the top edge.
const floatingControlsClassName: Record<FloatingControls, string> = {
  newLayout: "max-lg:top-2 max-lg:justify-center",
};

const chipButtonClassName = "size-9 shrink-0 rounded-full hover:bg-zinc-100";

interface Props {
  expertIdentity?: ExpertIdentity | null;
  /** The roster has not settled yet for an expert-scoped session. */
  isResolvingExpertIdentity?: boolean;
  /** Which layout floats its controls over the chat's top-left corner on
   *  small viewports; the chip lines up with them. */
  floatingControls?: FloatingControls;
}

/** Floats as a translucent chip pinned to the pane's top-left gutter — the
 *  zero-height wrapper keeps it out of the flex flow so messages scroll
 *  underneath. An expert session wears the expert's identity and every other
 *  session is Otto's, so the thread is never anonymous. The identity is a
 *  passive label; the expert's integrations live in the top-right controls.
 *  In the new layout on phones the chip also carries the sidebar button. */
export function ThreadHeader({
  expertIdentity,
  isResolvingExpertIdentity = false,
  floatingControls,
}: Props) {
  // While the roster loads, the chip shows a quiet placeholder rather than
  // Otto's identity, which would be wrong for an expert session.
  const isResolving = isResolvingExpertIdentity && !expertIdentity;
  const name = expertIdentity?.name ?? "Otto";
  const role = expertIdentity?.role ?? DEFAULT_EXPERT_ROLE;
  const jobTitle = expertIdentity?.jobTitle;
  const roleLabel = jobTitle || getExpertRoleLabel(role);
  const hasSidebar = Boolean(useAppSidebarToggle());
  const hasSidebarButton = floatingControls === "newLayout" && hasSidebar;

  return (
    <div data-testid="expert-thread-header" className="relative z-20 h-0">
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-3 flex justify-start px-4",
          floatingControls && floatingControlsClassName[floatingControls],
        )}
      >
        <div
          className={cn(
            "pointer-events-auto flex min-w-0 items-center gap-1 rounded-full border border-zinc-200/70 bg-white/75 py-1 pl-1.5 pr-5 shadow-sm backdrop-blur-md max-lg:shadow-md",
            hasSidebarButton && "max-md:pl-1",
          )}
        >
          {hasSidebarButton && (
            <AppSidebarTrigger
              className={cn(chipButtonClassName, "md:hidden")}
            />
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  tabIndex={0}
                  aria-label={
                    isResolving ? "Loading expert" : `${name} — ${roleLabel}`
                  }
                  className={cn(
                    "flex min-w-0 items-center gap-2 whitespace-nowrap rounded-full",
                    hasSidebarButton && "max-md:ml-1.5",
                  )}
                >
                  <ExpertAvatar
                    name={name}
                    avatarUrl={expertIdentity?.avatarUrl ?? null}
                    color={expertIdentity?.color}
                    isAutopilot={!expertIdentity && !isResolving}
                    isLoading={isResolving}
                    size="md"
                  />
                  {isResolving ? (
                    <Skeleton className="h-3.5 w-16 rounded" />
                  ) : (
                    <span className="min-w-0 max-w-[10rem]">
                      <ExpertIdentityDetails
                        isOtto={!expertIdentity}
                        name={name}
                        role={role}
                        jobTitle={jobTitle}
                        size="compact"
                      />
                    </span>
                  )}
                </div>
              </TooltipTrigger>
              {isResolving ? null : (
                <TooltipContent side="bottom">{roleLabel}</TooltipContent>
              )}
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </div>
  );
}
