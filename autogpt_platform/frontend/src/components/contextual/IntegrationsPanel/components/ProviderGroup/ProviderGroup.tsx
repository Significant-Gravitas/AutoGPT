"use client";

import { useState } from "react";
import Image from "next/image";
import { integrationIconSrc } from "@/components/molecules/IntegrationLogo/helpers";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import type { ProviderGroupView } from "../../helpers";
import { CredentialRow } from "../CredentialRow/CredentialRow";

interface Props {
  provider: ProviderGroupView;
  isSelected: (id: string) => boolean;
  onToggleSelected: (id: string) => void;
  onDelete: (id: string) => void;
  isDeletingId?: (id: string) => boolean;
}

export function ProviderGroup({
  provider,
  isSelected,
  onToggleSelected,
  onDelete,
  isDeletingId,
}: Props) {
  const count = provider.credentials.length;

  return (
    <Accordion
      type="single"
      collapsible
      defaultValue={provider.id}
      className="w-full overflow-hidden rounded-2xl border border-border-base bg-surface"
    >
      <AccordionItem value={provider.id} className="border-b-0">
        <AccordionTrigger className="px-3 py-3 pr-5 hover:no-underline [&>svg]:text-foreground-base">
          <div className="flex items-center gap-3">
            <ProviderAvatar provider={provider} />
            <span className="text-body-lg leading-6.5 font-medium tracking-title text-foreground-base">
              {provider.name}
            </span>
            <span className="inline-flex items-center justify-center rounded-lg bg-surface-hover px-2 py-0.5 text-body leading-5.5 font-medium text-foreground-base">
              {count}
            </span>
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-0 pt-0 pb-0">
          <div className="flex flex-col divide-y divide-border-base border-t border-border-base">
            {provider.credentials.map((credential) => (
              <CredentialRow
                key={credential.id}
                credential={credential}
                selected={isSelected(credential.id)}
                onToggleSelected={() => onToggleSelected(credential.id)}
                onDelete={() => onDelete(credential.id)}
                isDeleting={isDeletingId?.(credential.id) ?? false}
              />
            ))}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function ProviderAvatar({ provider }: { provider: ProviderGroupView }) {
  const [broken, setBroken] = useState(false);
  const src = provider.logoUrl ?? integrationIconSrc(provider.id);

  if (!src || broken) {
    return (
      <div
        aria-hidden="true"
        className="flex size-6 items-center justify-center rounded-full bg-surface-active text-tiny leading-inherit font-semibold text-foreground-body uppercase"
        data-testid={`provider-avatar-${provider.id}`}
      >
        {provider.name?.charAt(0) ?? provider.id.charAt(0)}
      </div>
    );
  }

  return (
    <Image
      src={src}
      alt={`${provider.name} logo`}
      width={24}
      height={24}
      onError={() => setBroken(true)}
      className="size-6 rounded-full bg-surface object-contain"
      unoptimized
    />
  );
}
