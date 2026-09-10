"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AUTOPILOT_AVATAR } from "@/components/molecules/BotAvatar/helpers";
import { ProviderBox } from "./components/ProviderBox";
import { useConnectStep } from "./useConnectStep";

// Subscriptions the experts could run on. Only ChatGPT has an adapter and
// provider approval today; the rest are named so the user knows what is
// coming, without claiming they work yet.
const UPCOMING = [
  { name: "Grok", logoSrc: "/integrations/xai.webp" },
  { name: "GitHub Copilot", logoSrc: "/integrations/github.png" },
];

/**
 * The last thing a self-host install asks for: a model to run on.
 *
 * The zero-config path (link the ChatGPT plan you already pay for) leads;
 * API keys are the advanced path and live in Settings. Deliberately
 * skippable so a user with keys is not blocked by a wizard.
 */
export function ConnectStep() {
  const { connect, isConnecting, skip, isAlreadyLinked } = useConnectStep();

  return (
    <FadeIn>
      <div className="flex w-full max-w-2xl flex-col items-center gap-8 px-4">
        <div className="flex flex-col items-center gap-4 text-center">
          <BotAvatar
            config={AUTOPILOT_AVATAR}
            status="idle"
            size={120}
            trackPointer
            showBadge={false}
          />
          <Text variant="h4" as="h1">
            Already paying for an AI subscription?
          </Text>
          <Text variant="large" as="p" tone="muted" className="max-w-md">
            Add it and your experts run on it. No API keys, no billing setup.
          </Text>
        </div>

        <div
          className="grid w-full grid-cols-1 gap-3 sm:grid-cols-3"
          aria-label="Subscriptions"
        >
          <ProviderBox
            name="ChatGPT"
            logoSrc="/integrations/openai.png"
            state={isAlreadyLinked ? "connected" : "available"}
            isBusy={isConnecting}
            onClick={connect}
          />
          {UPCOMING.map((provider) => (
            <ProviderBox
              key={provider.name}
              {...provider}
              state="coming-soon"
            />
          ))}
        </div>

        <Button size="small" onClick={skip} className="h-10 w-56 rounded-xl">
          Next
        </Button>
      </div>
    </FadeIn>
  );
}
