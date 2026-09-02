"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  KeyIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

import { ChoiceRow } from "./ChoiceRow";
import {
  isLinkedAccount,
  isSelectable,
  offerSubtitle,
  tierLabel,
  tierName,
  tierLock,
  tierSummary,
} from "./helpers";
import { nextRovingValue, rovingTabIndex } from "./radioKeys";
import { TierToggle } from "./TierToggle";
import { useConnectionPicker } from "./useConnectionPicker";

const TIERS = ["standard", "advanced"] as const;

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
}

/**
 * One control for what a turn runs on: the connection, and the quality tier
 * within it. Both come from the server-owned connection offer, so the client
 * decides nothing about routing, billing copy, or which models a tier maps to.
 */
export function ConnectionPicker({ connectionLocked = false }: Props) {
  const {
    offers,
    active,
    chooseConnection,
    tier,
    setTier,
    showTiers,
    hasConnectionChoice,
    isLoading,
    isError,
  } = useConnectionPicker();

  if (isLoading && offers.length === 0) return null;

  if (isError) {
    return (
      <span
        aria-label="AI connections unavailable"
        className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-destructive/20 bg-destructive/10 px-2.5 text-xs font-medium text-destructive"
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
        className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 text-xs font-medium text-foreground transition-colors hover:bg-muted/80"
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

  const connectionOptions = offers.map((offer) => ({
    value: offer.offer_id,
    disabled: !isSelectable(offer),
  }));
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
  // so and carries the accent that goes with it.
  const triggerLabel =
    runsOnOwnPlan && !showsTier ? `${label} · your plan` : label;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={
            showsTier
              ? `Model tier ${label} — change`
              : `Runs on ${triggerLabel} — change`
          }
          className={cn(
            "ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium shadow-sm transition-colors",
            runsOnOwnPlan && !showsTier
              ? "border-accent/40 bg-accent/5 text-accent hover:bg-accent/10"
              : "border-border bg-background text-foreground hover:bg-muted",
          )}
        >
          {/* The key stands for a credential. Against a tier name it would be
              labelling reasoning depth as an account, so it is dropped. */}
          {!showsTier && <Icon icon={KeyIcon} size={14} />}
          <span className="hidden sm:inline">{triggerLabel}</span>
          <Icon
            icon={ArrowDown01Icon}
            size={12}
            className="text-muted-foreground"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        align="start"
        className="w-[26rem] max-w-[calc(100vw-2rem)] border-border bg-popover p-0 text-popover-foreground"
      >
        {/* One runnable row is nothing to choose between. A locked-only row is
            different: the explanation and unlock link are the entire reason
            the chip remains visible. */}
        {(!connectionLocked || onlyOfferIsLocked) &&
          (offers.length > 1 || onlyOfferIsLocked) && (
            <>
              <SectionLabel>Runs on</SectionLabel>
              <div
                role="radiogroup"
                aria-label="Connection this chat runs on"
                onKeyDown={(event) => {
                  const to = nextRovingValue(
                    connectionOptions,
                    active?.offer_id ?? "",
                    event.key,
                  );
                  if (to === null) return;
                  event.preventDefault();
                  const target = offers.find((o) => o.offer_id === to);
                  if (target) chooseConnection(target);
                  event.currentTarget
                    .querySelector<HTMLElement>(`[data-offer="${to}"]`)
                    ?.focus();
                }}
              >
                {offers.map((offer) => (
                  <ChoiceRow
                    key={offer.offer_id}
                    offerId={offer.offer_id}
                    tabIndex={rovingTabIndex(
                      connectionOptions,
                      { value: offer.offer_id },
                      active?.offer_id ?? "",
                    )}
                    title={offer.display_name}
                    subtitle={offerSubtitle(offer)}
                    badge={isLinkedAccount(offer) ? "Connected" : undefined}
                    notes={[tierSummary(offer), ...offer.limitations].filter(
                      Boolean,
                    )}
                    isSelected={offer.offer_id === active?.offer_id}
                    onSelect={() => chooseConnection(offer)}
                    lock={
                      offer.lock_reason
                        ? {
                            reason: offer.lock_reason,
                            href: offer.unlock_href ?? null,
                          }
                        : undefined
                    }
                  />
                ))}
              </div>
            </>
          )}

        {showTiers && (
          <>
            <SectionLabel>Model tier</SectionLabel>
            <TierToggle
              value={tier}
              onSelect={setTier}
              segments={TIERS.map((candidate) => ({
                tier: candidate,
                label: tierLabel(active, candidate),
                lock: tierLock(active, candidate),
              }))}
            />
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3 pb-1 pt-3 text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
      {children}
    </p>
  );
}
