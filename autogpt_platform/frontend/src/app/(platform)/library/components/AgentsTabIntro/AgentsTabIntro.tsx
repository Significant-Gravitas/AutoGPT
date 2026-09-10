"use client";

import { TabIntroCard } from "@/app/(platform)/components/TabIntroCard/TabIntroCard";
import { useTabIntroCard } from "@/app/(platform)/components/TabIntroCard/useTabIntroCard";
import { GridViewIcon } from "@hugeicons/core-free-icons";

// First visit to the Agents tab. The fleet is already rendered behind the
// card, so "See my agents" only has to get out of the way.
export function AgentsTabIntro() {
  const { isOpen, dismiss, takeAction } = useTabIntroCard("agents");

  return (
    <TabIntroCard
      isOpen={isOpen}
      icon={GridViewIcon}
      title="Your mission control."
      body="What's running, what's scheduled, what needs you, and what it costs — all in one place."
      cta={{ label: "See my agents", onClick: () => takeAction("see_agents") }}
      onDismiss={dismiss}
    />
  );
}
