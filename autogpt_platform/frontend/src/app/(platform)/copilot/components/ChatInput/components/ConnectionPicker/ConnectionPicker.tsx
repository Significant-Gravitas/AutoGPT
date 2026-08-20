"use client";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import Link from "next/link";
import {
  PiCaretDown as CaretDownIcon,
  PiKey as KeyIcon,
  PiWarningCircle as WarningCircleIcon,
} from "react-icons/pi";

import { ChoiceRow } from "./ChoiceRow";
import { tierLabel, tierModel, tierName } from "./helpers";
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
        <WarningCircleIcon size={14} />
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
        <WarningCircleIcon size={14} />
        <span className="hidden sm:inline">Set up AI connection</span>
      </Link>
    );
  }

  // Nothing left to decide: either the tiers resolve to one model and the
  // connection is settled, or there is a single connection and its tiers are
  // the same model.
  if (!showTiers && (connectionLocked || offers.length === 1)) return null;

  const label = connectionLocked
    ? tierName(active, tier)
    : (active?.display_name ?? "Choose connection");

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={
            connectionLocked
              ? `Model tier ${label} — change`
              : `Runs on ${label} — change`
          }
          className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-background px-2.5 text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-muted"
        >
          <KeyIcon size={14} />
          <span className="hidden sm:inline">{label}</span>
          <CaretDownIcon size={12} className="text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start" className="w-80 p-0">
        {!connectionLocked && (
          <>
            <SectionLabel>Runs on</SectionLabel>
            <div role="radiogroup" aria-label="Connection this chat runs on">
              {offers.map((offer) => (
                <ChoiceRow
                  key={offer.offer_id}
                  title={offer.display_name}
                  subtitle={offer.backed_by_label}
                  notes={offer.limitations}
                  isSelected={offer.offer_id === active?.offer_id}
                  onSelect={() => chooseConnection(offer)}
                />
              ))}
            </div>
          </>
        )}

        {showTiers && (
          <>
            <SectionLabel>Model tier</SectionLabel>
            <div role="radiogroup" aria-label="Model tier">
              {TIERS.map((candidate) => (
                <ChoiceRow
                  key={candidate}
                  title={tierName(active, candidate)}
                  subtitle={tierModel(active, candidate) ?? undefined}
                  label={tierLabel(active, candidate)}
                  isSelected={tier === candidate}
                  onSelect={() => setTier(candidate)}
                />
              ))}
            </div>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3 pb-1 pt-3 text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
      {children}
    </p>
  );
}
