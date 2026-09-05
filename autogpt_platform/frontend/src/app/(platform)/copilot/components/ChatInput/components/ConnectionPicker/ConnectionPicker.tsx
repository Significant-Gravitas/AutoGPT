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
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  AiBrain01Icon,
  CloudServerIcon,
  FlashIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";

import { ChoiceRow } from "./ChoiceRow";
import { ConnectAccountRow } from "./ConnectAccountRow";
import { Swap } from "./Swap";
import {
  isLinkedAccount,
  isSelectable,
  offerSubtitle,
  tierLabel,
  tierModel,
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
  const {
    offers,
    active,
    chooseConnection,
    tier,
    setTier,
    showTiers,
    hasConnectionChoice,
    connectChatGPT,
    isConnecting,
    canConnectChatGPT,
    isLoading,
    isError,
  } = useConnectionPicker();

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
  // so.
  const triggerLabel =
    runsOnOwnPlan && !showsTier ? `${label} · your plan` : label;
  // One runnable row is nothing to choose between. A locked-only row is
  // different: the explanation and unlock link are the entire reason the chip
  // remains visible.
  const showsConnections =
    (!connectionLocked || onlyOfferIsLocked) &&
    (offers.length > 1 || onlyOfferIsLocked);

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
        className="w-[24rem] max-w-[calc(100vw-2rem)] rounded-2xl border-zinc-200 bg-[#F9F9FA] p-3 pt-4 text-zinc-900 shadow-lg"
      >
        {showsConnections && (
          <>
            <SectionLabel>Runs on</SectionLabel>
            <div
              role="radiogroup"
              aria-label="Connection this chat runs on"
              // The connections read as plain rows on the sheet: a card around
              // them would frame the choice twice, once here and once around
              // the tiers below.
              className="overflow-hidden rounded-xl"
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
                  leading={<OfferMark offer={offer} />}
                  title={offer.display_name}
                  subtitle={offerSubtitle(offer)}
                  badge={isLinkedAccount(offer) ? "Connected" : undefined}
                  // A selectable row's models are named by the tier toggle
                  // right below it, so repeating them here only crowds the
                  // choice. A locked row has no toggle, so it keeps them.
                  notes={
                    offer.lock_reason
                      ? [tierSummary(offer), ...offer.limitations].filter(
                          Boolean,
                        )
                      : offer.limitations
                  }
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
            <div className="overflow-hidden rounded-xl border border-neutral-200 bg-white">
              <TierToggle
                value={tier}
                onSelect={setTier}
                segments={TIERS.map((candidate) => ({
                  tier: candidate,
                  label: tierLabel(active, candidate),
                  name: tierName(active, candidate),
                  model: tierModel(active, candidate),
                  lock: tierLock(active, candidate),
                }))}
              />
            </div>
          </>
        )}

        {canConnectChatGPT && (
          <>
            <SectionLabel>Add a connection</SectionLabel>
            <ConnectAccountRow
              onConnect={connectChatGPT}
              isConnecting={isConnecting}
            />
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}

/**
 * What a connection is, before its name is read: the provider's own logo where
 * there is one, and the machine it runs on where the route is this deployment.
 */
function OfferMark({ offer }: { offer: AIConnectionOffer }) {
  if (offer.auth_method === "deployment") {
    return <Icon icon={CloudServerIcon} size={20} className="text-zinc-500" />;
  }
  return <IntegrationLogo provider={offer.provider_family} size={20} />;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3 pb-1.5 pt-3 text-[11px] font-medium uppercase tracking-[0.06em] text-zinc-500 first:pt-0">
      {children}
    </p>
  );
}
