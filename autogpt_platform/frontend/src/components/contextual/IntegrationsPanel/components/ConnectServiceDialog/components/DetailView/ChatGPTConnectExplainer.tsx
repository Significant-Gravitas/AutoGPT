"use client";

import {
  ArrowDataTransferHorizontalIcon,
  ShieldKeyIcon,
  SparklesIcon,
  Target02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

import { useGetV2ListProviderModelTiers } from "@/app/api/__generated__/endpoints/chat/chat";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

import { chatgptModelsSentence } from "./helpers";

interface Point {
  icon: IconSvgElement;
  title: string;
  body: string;
  withModels?: (models: string) => string;
}

/**
 * Linking a ChatGPT plan changes who pays for a run, so the four questions
 * that implies get answered on screen before the OAuth window opens rather
 * than after the user has committed.
 *
 * Says nothing about pausing and resuming a run at a provider limit, because
 * same-chat recovery is not built yet.
 *
 * It does name the models, from the catalog rather than from a literal here.
 * The connections list cannot answer that at this point -- the user has not
 * made this connection, so it is absent from their offers -- which is what
 * the provider-tiers endpoint exists for.
 */
const POINTS: Point[] = [
  {
    icon: ArrowDataTransferHorizontalIcon,
    title: "What it does.",
    body: "Chats routed to ChatGPT run on your ChatGPT plan instead of AutoGPT credits. Every new chat starts on the connection you choose in Settings, and you can change it per conversation — nothing switches on its own.",
  },
  {
    icon: Target02Icon,
    title: "What it costs.",
    body: "Usage counts toward your ChatGPT plan's own limits. Those runs spend zero AutoGPT credits, and AutoGPT never adds a charge on top.",
  },
  {
    icon: SparklesIcon,
    title: "What you get.",
    body: "The models your ChatGPT plan already includes, at no extra cost from AutoGPT.",
    /** Replaced with the named models when the catalog can supply them. */
    withModels: (models: string) => `${models}, at no extra cost from AutoGPT.`,
  },
  {
    icon: ShieldKeyIcon,
    title: "Stay in control.",
    body: "A chat stays on the connection it started with, so a run is never moved onto your AutoGPT credits without asking. Disconnect any time in Settings.",
  },
];

export function ChatGPTConnectExplainer() {
  const { data } = useGetV2ListProviderModelTiers({
    query: { refetchOnWindowFocus: false },
  });
  const models =
    data?.status === 200 ? chatgptModelsSentence(data.data.providers) : "";

  return (
    <div className="divide-y divide-[#DADADC] rounded-2xl border border-[#DADADC] bg-[#FBFBFC]">
      {POINTS.map((point) => (
        <div key={point.title} className="flex items-start gap-3 p-4">
          <span
            aria-hidden
            className="mt-[2px] flex h-6 w-6 flex-none items-center justify-center rounded-lg bg-[#F1EBFF] text-[#7444E5]"
          >
            <Icon icon={point.icon} size={14} />
          </span>
          <Text variant="small" className="text-[#505057]">
            <span className="font-medium text-black">{point.title}</span>{" "}
            {models && point.withModels ? point.withModels(models) : point.body}
          </Text>
        </div>
      ))}
    </div>
  );
}
