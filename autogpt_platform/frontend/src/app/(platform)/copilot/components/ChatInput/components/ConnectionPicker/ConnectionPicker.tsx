"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  AiBrain01Icon,
  FlashIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

import { ConnectionPickerPanel } from "./ConnectionPickerPanel";
import { Swap } from "./Swap";
import {
  isLinkedAccount,
  listedConnections,
  tierLabel,
  tierLock,
  tierName,
} from "./helpers";
import { useConnectionPicker } from "./useConnectionPicker";

interface Props {
  /**
   * The connection is settled and only the tier is still open.
   *
   * A session's connection is chosen when it is created and fixed for its
   * lifetime, but the tier is a per-message setting -- it can change between
   * turns of a chat already underway. Hiding the whole control once a session
   * exists would take that away.
   */
  connectionLocked?: boolean;
  className?: string;
}

/**
 * One control for what a turn runs on: the connection, and the quality tier
 * within it. Both come from the server-owned connection offer, so the client
 * decides nothing about routing, billing copy, or which models a tier maps to.
 */
export function ConnectionPicker({
  connectionLocked = false,
  className,
}: Props) {
  const picker = useConnectionPicker();
  const {
    offers,
    active,
    tier,
    showTiers,
    hasConnectionChoice,
    isLoading,
    isError,
  } = picker;
  const advancedLock = tierLock(active, "advanced");
  const showMaxUpgrade =
    active?.auth_method === "deployment" && Boolean(advancedLock);

  if (isLoading && offers.length === 0) return null;

  if (isError) {
    return (
      <span
        aria-label="AI connections unavailable"
        className={cn(
          "inline-flex h-9 items-center gap-1.5 rounded-full border border-destructive/20 bg-destructive/10 px-2.5 text-xs font-medium text-destructive",
          className,
        )}
      >
        <Icon icon={AlertCircleIcon} size={14} />
        <span className="hidden sm:inline">Connections unavailable</span>
      </span>
    );
  }

  if (offers.length === 0) {
    return (
      <Link
        href="/settings/integrations"
        aria-label="Set up an AI connection"
        className={cn(
          "inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 text-xs font-medium text-foreground transition-colors hover:bg-muted/80",
          className,
        )}
      >
        <Icon icon={AlertCircleIcon} size={14} />
        <span className="hidden sm:inline">Set up AI connection</span>
      </Link>
    );
  }

  // Nothing left to decide when the one offer can actually run. A locked-only
  // offer still needs this control so the user can see why it is unavailable
  // and follow its unlock link.
  const onlyOfferIsLocked =
    offers.length === 1 && Boolean(offers[0]?.lock_reason);
  if (
    !showTiers &&
    !onlyOfferIsLocked &&
    (connectionLocked || offers.length === 1)
  ) {
    return null;
  }

  const runsOnOwnPlan = active ? isLinkedAccount(active) : false;
  // With no connection to choose between, naming it says nothing — a hosted
  // user would read "AutoGPT Platform" on every chat and learn only that there
  // is one. The tier is the live choice, so that is what the chip carries.
  // A second connection makes the route itself a decision, and the label
  // switches to it.
  // A tier only names a route that can actually run. With a locked-only offer
  // there is no active connection, so the chip must stay a connection prompt
  // and open the explanation instead of claiming a phantom Balanced tier.
  const showsTier =
    (connectionLocked && Boolean(active)) ||
    (!hasConnectionChoice && Boolean(active));
  const label = showsTier
    ? tierName(active, tier)
    : (active?.display_name ?? "Choose connection");
  // "ChatGPT · your plan". Spending your own subscription rather than the
  // platform's is the thing worth noticing before sending, so the chip says
  // so.
  const triggerLabel =
    runsOnOwnPlan && !showsTier ? `${label} · your plan` : label;

  // Naming only the tier, the chip folds down to its glyph among the other
  // quiet icons on the composer's right; the tier and its model wait in the
  // tooltip. A connection name is a decision the user has to read, so that
  // form keeps its label.
  const trigger = (
    <PopoverTrigger asChild>
      <Button
        type="button"
        variant="ghost"
        size={showsTier ? "icon" : "small"}
        unmask={false}
        aria-label={
          showsTier
            ? `Model tier ${label} — change`
            : `Runs on ${triggerLabel} — change`
        }
        className={cn(
          showsTier
            ? "size-8 p-0 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700"
            : "h-9 min-w-0 gap-1.5 px-2.5 py-1 text-sm",
          className,
        )}
      >
        {/* The tier the next turn runs at, whichever half of the setting
            the label happens to be naming. It swaps on the same beat as the
            label beside it, so the two read as one change. */}
        <Swap swapKey={tier} className="flex-none">
          <Icon
            icon={tier === "advanced" ? AiBrain01Icon : FlashIcon}
            size={16}
          />
        </Swap>
        {!showsTier && (
          <>
            <span className="hidden sm:inline">
              <Swap>{triggerLabel}</Swap>
            </span>
            <Icon
              icon={ArrowDown01Icon}
              size={14}
              className="text-muted-foreground"
            />
          </>
        )}
      </Button>
    </PopoverTrigger>
  );

  return (
    <Popover>
      {showsTier ? (
        <Tooltip>
          <TooltipTrigger asChild>{trigger}</TooltipTrigger>
          <TooltipContent side="top">{tierLabel(active, tier)}</TooltipContent>
        </Tooltip>
      ) : (
        trigger
      )}

      <PopoverContent
        align="end"
        className={cn(
          "max-h-[var(--radix-popover-content-available-height)] w-[24rem] max-w-[calc(100vw-2rem)] overflow-y-auto rounded-2xl border-zinc-200 bg-[#F9F9FA] p-3 pt-4 text-zinc-900 shadow-lg",
          showMaxUpgrade && "bg-white",
        )}
      >
        <ConnectionPickerPanel
          picker={picker}
          rows={listedConnections(offers, connectionLocked)}
        />
      </PopoverContent>
    </Popover>
  );
}
