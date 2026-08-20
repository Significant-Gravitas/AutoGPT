"use client";

import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import Link from "next/link";
import {
  PiCaretDown as CaretDownIcon,
  PiKey as KeyIcon,
  PiWarningCircle as WarningCircleIcon,
} from "react-icons/pi";

import { tierLabel } from "./helpers";
import { useConnectionPicker } from "./useConnectionPicker";

const TIERS = ["standard", "advanced"] as const;

/**
 * One control for what a turn runs on: the connection, and the quality tier
 * within it. Both come from the server-owned connection offer, so the client
 * decides nothing about routing, billing copy, or which models a tier maps to.
 */
export function ConnectionPicker() {
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

  // One connection whose tiers are the same model is not a choice at all.
  if (offers.length === 1 && !showTiers) return null;

  const label = active?.display_name ?? "Choose connection";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`Runs on ${label} — change`}
          className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-background px-2.5 text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-muted"
        >
          <KeyIcon size={14} />
          <span className="hidden sm:inline">{label}</span>
          <CaretDownIcon size={12} className="text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start" className="w-80 p-0">
        <SectionLabel>Runs on</SectionLabel>
        <div role="radiogroup" aria-label="Connection this chat runs on">
          {offers.map((offer) => (
            <OfferRow
              key={offer.offer_id}
              offer={offer}
              isSelected={offer.offer_id === active?.offer_id}
              onSelect={() => chooseConnection(offer)}
            />
          ))}
        </div>

        {showTiers && (
          <>
            <SectionLabel>Model tier</SectionLabel>
            <div
              role="radiogroup"
              aria-label="Model tier"
              className="flex gap-1 p-2 pt-0"
            >
              {TIERS.map((candidate) => (
                <button
                  key={candidate}
                  type="button"
                  role="radio"
                  aria-checked={tier === candidate}
                  onClick={() => setTier(candidate)}
                  className={cn(
                    "flex-1 truncate rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    tier === candidate
                      ? "bg-background text-foreground shadow-sm ring-1 ring-border"
                      : "text-muted-foreground hover:bg-muted",
                  )}
                >
                  {tierLabel(active, candidate)}
                </button>
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

interface OfferRowProps {
  offer: AIConnectionOffer;
  isSelected: boolean;
  onSelect: () => void;
}

function OfferRow({ offer, isSelected, onSelect }: OfferRowProps) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      onClick={onSelect}
      className={cn(
        "flex w-full items-start gap-2.5 px-3 py-2 text-left transition-colors",
        "focus-visible:bg-muted focus-visible:outline-none",
        isSelected ? "bg-muted/60" : "hover:bg-muted/40",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "mt-[3px] flex h-3.5 w-3.5 flex-none items-center justify-center rounded-full border",
          isSelected ? "border-primary" : "border-muted-foreground/50",
        )}
      >
        {isSelected && (
          <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
        )}
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="text-xs font-medium text-foreground">
          {offer.display_name}
        </span>
        <span className="text-[11px] leading-snug text-muted-foreground">
          {offer.backed_by_label}
        </span>
        {offer.limitations.map((limitation) => (
          <span
            key={limitation}
            className="mt-0.5 text-[11px] leading-snug text-muted-foreground/80"
          >
            {limitation}
          </span>
        ))}
      </span>
    </button>
  );
}
