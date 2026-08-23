"use client";

import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  icon: IconSvgElement;
  count?: number;
  action?: ReactNode;
  children: ReactNode;
}

/**
 * A titled card in the chat's floating stack — label outside, content in the
 * card, mirroring the home briefing tiles (``HomeTile``). Keeping the title
 * out of the card is what separates one section from the next.
 */
export function StackSection({ title, icon, count, action, children }: Props) {
  return (
    <section className="flex min-w-0 flex-col">
      {/* px-4 matches the card's own padding, so the title starts exactly
          where the row content inside does. */}
      <div className="mb-1.5 flex items-center gap-1.5 px-4">
        <Icon icon={icon} size={14} className="text-zinc-500" aria-hidden />
        <Text variant="small-medium" className="!text-zinc-700">
          {title}
          {count === undefined ? "" : ` (${count})`}
        </Text>
        {action && <div className="ml-auto flex items-center">{action}</div>}
      </div>
      <div className="rounded-3xl bg-white/90 px-4 py-3 backdrop-blur smooth-shadow-ring-sm">
        {children}
      </div>
    </section>
  );
}
