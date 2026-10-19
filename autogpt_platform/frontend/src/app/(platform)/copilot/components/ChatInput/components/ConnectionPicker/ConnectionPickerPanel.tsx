"use client";

import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { Icon } from "@/components/atoms/Icon/Icon";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { CloudServerIcon } from "@hugeicons/core-free-icons";

import { ChoiceRow } from "./ChoiceRow";
import { ConnectAccountRow } from "./ConnectAccountRow";
import {
  isLinkedAccount,
  isSelectable,
  offerSubtitle,
  tierLabel,
  tierLock,
  tierModel,
  tierName,
  tierSummary,
} from "./helpers";
import { MaxUpgradeCard } from "./MaxUpgradeCard";
import { nextRovingValue, rovingTabIndex } from "./radioKeys";
import { TierToggle } from "./TierToggle";
import type { useConnectionPicker } from "./useConnectionPicker";

const TIERS = ["standard", "advanced"] as const;

interface Props {
  picker: ReturnType<typeof useConnectionPicker>;
  /** The connections listed under "Runs on"; empty hides the section. */
  rows: AIConnectionOffer[];
}

/** The picker's sheet: which connection a turn runs on and at what tier. */
export function ConnectionPickerPanel({ picker, rows }: Props) {
  const {
    active,
    chooseConnection,
    tier,
    setTier,
    showTiers,
    connectChatGPT,
    isConnecting,
    canConnectChatGPT,
  } = picker;
  const advancedLock = tierLock(active, "advanced");
  const showMaxUpgrade =
    active?.auth_method === "deployment" && Boolean(advancedLock);
  const connectionOptions = rows.map((offer) => ({
    value: offer.offer_id,
    disabled: !isSelectable(offer),
  }));

  return (
    <>
      {rows.length > 0 && (
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
              // `event.key` goes straight into the roving helper, so guard
              // here instead.
              if (isComposingEvent(event)) return;
              const to = nextRovingValue(
                connectionOptions,
                active?.offer_id ?? "",
                event.key,
              );
              if (to === null) return;
              event.preventDefault();
              const target = rows.find((o) => o.offer_id === to);
              if (target) chooseConnection(target);
              event.currentTarget
                .querySelector<HTMLElement>(`[data-offer="${to}"]`)
                ?.focus();
            }}
          >
            {rows.map((offer) => (
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
                    ? [tierSummary(offer), ...offer.limitations].filter(Boolean)
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
          <div
            className={cn(
              "overflow-hidden rounded-xl border border-neutral-200 bg-white",
              showMaxUpgrade && "border-0",
            )}
          >
            <TierToggle
              value={tier}
              onSelect={setTier}
              advancedUpgrade={
                showMaxUpgrade && advancedLock?.href ? (
                  <MaxUpgradeCard
                    label={tierLabel(active, "advanced")}
                    name={tierName(active, "advanced")}
                    model={tierModel(active, "advanced")}
                    reason={advancedLock.reason}
                    href={advancedLock.href}
                  />
                ) : undefined
              }
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
        <div
          className={cn(
            "mt-4",
            showMaxUpgrade && "border-t border-zinc-200 pt-4",
          )}
        >
          <SectionLabel>Add a connection</SectionLabel>
          <ConnectAccountRow
            onConnect={connectChatGPT}
            isConnecting={isConnecting}
          />
        </div>
      )}
    </>
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
