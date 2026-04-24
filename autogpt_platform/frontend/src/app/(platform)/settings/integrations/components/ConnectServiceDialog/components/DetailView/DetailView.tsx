"use client";

import { useState } from "react";
import Image from "next/image";
import { ArrowLeftIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";

import { ConnectableProvider } from "../../helpers";
import { ApiKeyConnectForm } from "./ApiKeyConnectForm";
import { OAuthConnectButton } from "./OAuthConnectButton";

interface Props {
  provider: ConnectableProvider;
  onBack: () => void;
  onSuccess: () => void;
}

export function DetailView({ provider, onBack, onSuccess }: Props) {
  const description = provider.description;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-3">
        <Button
          variant="icon"
          size="icon"
          onClick={onBack}
          aria-label="Back to services"
          className="size-9"
          withTooltip={false}
        >
          <ArrowLeftIcon size={18} />
        </Button>
        <ProviderAvatar id={provider.id} name={provider.name} />
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="h4" as="h2" className="text-[#1F1F20]">
            {provider.name}
          </Text>
          {description ? (
            <Text variant="small" className="truncate text-[#83838C]">
              {description}
            </Text>
          ) : null}
        </div>
      </div>

      <TabsLine defaultValue="oauth">
        <TabsLineList>
          <TabsLineTrigger value="oauth">OAuth</TabsLineTrigger>
          <TabsLineTrigger value="api_key">API key</TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent value="oauth">
          <OAuthConnectButton
            provider={provider.id}
            providerName={provider.name}
            onSuccess={onSuccess}
          />
        </TabsLineContent>
        <TabsLineContent value="api_key">
          <ApiKeyConnectForm
            provider={provider.id}
            providerName={provider.name}
            onSuccess={onSuccess}
          />
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}

function ProviderAvatar({ id, name }: { id: string; name: string }) {
  const [broken, setBroken] = useState(false);
  if (broken) {
    return <div aria-hidden className="size-10 shrink-0 rounded-md bg-zinc-100" />;
  }
  return (
    <Image
      src={`/integrations/${id}.png`}
      alt={`${name} logo`}
      width={40}
      height={40}
      className="size-10 shrink-0 object-contain"
      onError={() => setBroken(true)}
      unoptimized
    />
  );
}
