import type { ReactNode } from "react";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  number: number;
  title: string;
  intro: string;
  children: ReactNode;
}

export function UsageStep({ number, title, intro, children }: Props) {
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-900 text-xs font-medium text-white">
          {number}
        </span>
        <Text variant="body-medium" as="h3" className="text-textBlack">
          {title}
        </Text>
      </div>
      <Text variant="small" className="text-zinc-600">
        {intro}
      </Text>
      {children}
    </section>
  );
}
