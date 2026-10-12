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
import { cn } from "@/lib/utils";
import type { ExpertIdentity } from "../../../useExpertMap";
import { ExpertAvatar } from "./ExpertAvatar/ExpertAvatar";

// Otto's product-facing title when a session carries no expert identity.
const DEFAULT_EXPERT_ROLE = "Head of AI";

export type FloatingControls = "classic" | "newLayout";

// Below `lg` the chip joins the floating controls' row: centred on their
// height, one button gap after the last one. Classic: MobileHeader's 46px
// buttons sit 20px down and 16px in; the chat column starts 8px in. New
// layout: the inset header's 36px buttons sit 24px down and in (32px in from
// `md`, where only the files button shows).
const floatingControlsClassName: Record<FloatingControls, string> = {
  classic: "max-lg:top-5 max-lg:pl-[116px]",
  newLayout: "max-lg:top-[19px] max-md:pl-[104px] md:max-lg:pl-[68px]",
};

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
 *  session is Otto's, so the thread is never anonymous. The chip is a passive
 *  label; the expert's integrations live in the top-right controls. */
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

  return (
    <div data-testid="expert-thread-header" className="relative z-20 h-0">
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-3 flex justify-start px-4",
          floatingControls && floatingControlsClassName[floatingControls],
        )}
      >
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                tabIndex={0}
                aria-label={
                  isResolving ? "Loading expert" : `${name} — ${roleLabel}`
                }
                className="pointer-events-auto flex min-w-0 items-center gap-2 whitespace-nowrap rounded-full border border-zinc-200/70 bg-white/75 py-1 pl-1.5 pr-5 shadow-sm backdrop-blur-md max-lg:shadow-md"
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
              <TooltipContent
                side="bottom"
                className="bg-zinc-900 text-zinc-50 outline-none"
              >
                {roleLabel}
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}
