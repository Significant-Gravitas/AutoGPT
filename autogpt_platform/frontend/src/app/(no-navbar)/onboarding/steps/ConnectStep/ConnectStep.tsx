"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Text } from "@/components/atoms/Text/Text";
import { DeviceAuthConnectButton } from "@/components/contextual/DeviceAuth/DeviceAuthConnectButton";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AUTOPILOT_AVATAR } from "@/components/molecules/BotAvatar/helpers";
import { ProviderBox } from "./components/ProviderBox";
import { useConnectStep } from "./useConnectStep";

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
  const {
    connect,
    finishConnection,
    isConnecting,
    skip,
    isChatGPTLinked,
    isMicrosoftLinked,
  } = useConnectStep();

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
            Connect it for AutoGPT chats. No API keys, no billing setup.
          </Text>
        </div>

        <div
          className="grid w-full grid-cols-1 gap-3 sm:grid-cols-3"
          aria-label="Subscriptions"
        >
          <ProviderBox
            name="ChatGPT"
            logoSrc="/integrations/openai.png"
            state={isChatGPTLinked ? "connected" : "available"}
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

        <div className="w-full rounded-2xl border border-zinc-200 bg-white p-4">
          {isMicrosoftLinked ? (
            <Text variant="body" as="p">
              Your Microsoft 365 Copilot is connected. It can answer chats but
              does not run AutoGPT tools.
            </Text>
          ) : (
            <>
              <DeviceAuthConnectButton
                provider="microsoft_365_copilot"
                providerName="Microsoft 365 Copilot"
                onSuccess={finishConnection}
              />
              <Text variant="small" as="p" tone="muted" className="mt-3">
                Requires a paid Microsoft Copilot or Copilot Business add-on
                from your work or school organization. The included Microsoft
                365 Copilot Chat does not qualify. This connection answers chats
                but does not run AutoGPT tools.
              </Text>
            </>
          )}
        </div>

        <Button size="small" onClick={skip} className="h-10 w-56 rounded-xl">
          Next
        </Button>
      </div>
    </FadeIn>
  );
}
