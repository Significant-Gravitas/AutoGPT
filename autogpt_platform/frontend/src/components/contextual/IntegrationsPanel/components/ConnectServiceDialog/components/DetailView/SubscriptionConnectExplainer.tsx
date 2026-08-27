"use client";

import {
  ArrowDataTransferHorizontalIcon,
  ShieldKeyIcon,
  SparklesIcon,
  Target02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Point {
  icon: IconSvgElement;
  title: string;
  body: string;
  withModels?: (models: string) => string;
}

/**
 * Linking a subscription changes who pays for a run, so the four questions
 * that implies get answered on screen before the sign-in window opens rather
 * than after the user has committed.
 *
 * The provider is named rather than assumed: this panel used to exist only
 * for ChatGPT and said so in every sentence, which made a second provider a
 * rewrite instead of a prop.
 *
 * Says nothing about pausing and resuming a run at a provider limit, because
 * same-chat recovery is not built yet.
 *
 * It does name the models, from the catalog rather than from a literal here.
 * The connections list cannot answer that at this point -- the user has not
 * made this connection, so it is absent from their offers -- which is what
 * the provider-tiers endpoint exists for.
 */
function pointsFor(providerName: string): Point[] {
  return [
    {
      icon: ArrowDataTransferHorizontalIcon,
      title: "What it does.",
      body: `Chats routed to ${providerName} run on your ${providerName} subscription instead of AutoGPT credits. Every new chat starts on the connection you choose in Settings, and you can change it per conversation — nothing switches on its own.`,
    },
    {
      icon: Target02Icon,
      title: "What it costs.",
      body: `Usage counts toward your ${providerName} subscription's own limits. Those runs spend zero AutoGPT credits, and AutoGPT never adds a charge on top.`,
    },
    {
      icon: SparklesIcon,
      title: "What you get.",
      body: `The models your ${providerName} subscription already includes, at no extra cost from AutoGPT.`,
      /** Replaced with the named models when the catalog can supply them. */
      withModels: (models: string) =>
        `${models}, at no extra cost from AutoGPT.`,
    },
    {
      icon: ShieldKeyIcon,
      title: "Stay in control.",
      body: "A chat stays on the connection it started with, so a run is never moved onto your AutoGPT credits without asking. Disconnect any time in Settings.",
    },
  ];
}

interface Props {
  providerName: string;
  modelsSentence: string;
}

export function SubscriptionConnectExplainer({
  providerName,
  modelsSentence,
}: Props) {
  return (
    <div className="divide-y divide-[#DADADC] rounded-2xl border border-[#DADADC] bg-[#FBFBFC]">
      {pointsFor(providerName).map((point) => (
        <div key={point.title} className="flex items-start gap-3 p-4">
          <span
            aria-hidden
            className="mt-[2px] flex h-6 w-6 flex-none items-center justify-center rounded-lg bg-[#F1EBFF] text-[#7444E5]"
          >
            <Icon icon={point.icon} size={14} />
          </span>
          <Text variant="small" className="text-[#505057]">
            <span className="font-medium text-black">{point.title}</span>{" "}
            {modelsSentence && point.withModels
              ? point.withModels(modelsSentence)
              : point.body}
          </Text>
        </div>
      ))}
    </div>
  );
}
