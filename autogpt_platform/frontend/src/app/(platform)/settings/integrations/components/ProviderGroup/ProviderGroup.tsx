"use client";

import { useState } from "react";
import Image from "next/image";
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
      className="w-full overflow-hidden rounded-lg border border-[#DADADC] bg-white"
    >
      <AccordionItem value={provider.id} className="border-b-0">
        <AccordionTrigger className="px-3 py-3 pr-5 hover:no-underline [&>svg]:text-[#1F1F20]">
          <div className="flex items-center gap-3">
            <ProviderAvatar provider={provider} />
            <span className="text-[16px] font-medium leading-[26px] tracking-[-0.08px] text-black">
              {provider.name}
            </span>
            <span className="inline-flex items-center justify-center rounded-[10px] bg-[#EFF1F4] px-2 py-[2px] text-[14px] font-medium leading-[22px] text-black">
              {count}
            </span>
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-0 pb-0 pt-0">
          <div className="flex flex-col divide-y divide-[#DADADC] border-t border-[#DADADC]">
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
  const src = provider.logoUrl ?? `/integrations/${provider.id}.png`;

  if (broken) {
    return (
      <div
        aria-hidden="true"
        className="flex size-6 items-center justify-center rounded-full bg-[#D9D9D9] text-[10px] font-semibold uppercase text-zinc-700"
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
      className="size-6 rounded-full bg-white object-contain"
      unoptimized
    />
  );
}
