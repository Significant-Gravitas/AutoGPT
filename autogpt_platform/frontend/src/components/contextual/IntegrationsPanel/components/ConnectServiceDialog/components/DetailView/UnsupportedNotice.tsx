"use client";
import { Text } from "@/components/atoms/Text/Text";
import { Plug01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  providerName: string;
  detail?: string;
}

export function UnsupportedNotice({ providerName, detail }: Props) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-zinc-200 px-6 py-10 text-center">
      <Icon icon={Plug01Icon} size={28} className="text-muted-foreground" />
      <Text variant="body">No connection method available</Text>
      <Text variant="small" tone="muted" className="max-w-[360px]">
        {detail ??
          `${providerName} doesn't currently expose a connection flow you can manage from settings. Add a block that uses ${providerName} to enable this.`}
      </Text>
    </div>
  );
}
