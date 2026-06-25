"use client";

import { PlugIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  providerName: string;
  detail?: string;
}

export function UnsupportedNotice({ providerName, detail }: Props) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-[#DADADC] px-6 py-10 text-center">
      <PlugIcon size={28} className="text-[#83838C]" />
      <Text variant="body" className="text-[#1F1F20]">
        No connection method available
      </Text>
      <Text variant="small" className="max-w-[360px] text-[#83838C]">
        {detail ??
          `${providerName} doesn't currently expose a connection flow you can manage from settings. Add a block that uses ${providerName} to enable this.`}
      </Text>
    </div>
  );
}
