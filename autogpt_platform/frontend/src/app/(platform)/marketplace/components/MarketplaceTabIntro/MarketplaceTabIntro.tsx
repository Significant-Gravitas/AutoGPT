"use client";

import { TabIntroCard } from "@/app/(platform)/components/TabIntroCard/TabIntroCard";
import { useTabIntroCard } from "@/app/(platform)/components/TabIntroCard/useTabIntroCard";
import { Store01Icon } from "@hugeicons/core-free-icons";
import { AGENTS_SECTION_ID, FEATURED_SECTION_ID } from "./helpers";

// First visit to the Marketplace tab. The CTA lands the user on the
// hand-picked carousel rather than the top of a long page, falling back to the
// full listing on the days nothing is featured — the card promises movement,
// so it has to produce some.
export function MarketplaceTabIntro() {
  const { isOpen, dismiss, takeAction } = useTabIntroCard("marketplace");

  function browseFeatured() {
    takeAction("browse_featured");
    const target =
      document.getElementById(FEATURED_SECTION_ID) ??
      document.getElementById(AGENTS_SECTION_ID);
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <TabIntroCard
      isOpen={isOpen}
      icon={Store01Icon}
      title="Agents ready to work."
      body="Hundreds built by the community. Install one in a single click and run it today."
      cta={{ label: "Browse featured agents", onClick: browseFeatured }}
      onDismiss={dismiss}
    />
  );
}
