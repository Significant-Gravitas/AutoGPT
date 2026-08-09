"use client";

import { TabIntroCard } from "@/app/(platform)/components/TabIntroCard/TabIntroCard";
import { useTabIntroCard } from "@/app/(platform)/components/TabIntroCard/useTabIntroCard";
import { Store01Icon } from "@hugeicons/core-free-icons";
import { FEATURED_SECTION_ID } from "../FeaturedSection/FeaturedSection";

// First visit to the Marketplace tab. The CTA lands the user on the
// hand-picked carousel rather than the top of a long page; if there is
// nothing featured right now it just gets out of the way.
export function MarketplaceTabIntro() {
  const { isOpen, dismiss, takeAction } = useTabIntroCard("marketplace");

  function browseFeatured() {
    takeAction("browse_featured");
    document
      .getElementById(FEATURED_SECTION_ID)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
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
