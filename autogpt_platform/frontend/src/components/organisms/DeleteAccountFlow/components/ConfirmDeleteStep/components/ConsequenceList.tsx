import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CreditCardIcon,
  Delete02Icon,
  Store01Icon,
  Unlink01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { Consequence, ConsequenceKey } from "../helpers";

interface Props {
  items: Consequence[];
}

const CONSEQUENCE_ICON: Record<ConsequenceKey, IconSvgElement> = {
  data: Delete02Icon,
  apps: Unlink01Icon,
  plan: CreditCardIcon,
  listings: Store01Icon,
};

export function ConsequenceList({ items }: Props) {
  return (
    <ul className="flex flex-col divide-y divide-separator rounded-xl border border-border-base">
      {items.map((item) => (
        <li key={item.key} className="flex items-start gap-3 px-4 py-3">
          <Icon
            icon={CONSEQUENCE_ICON[item.key]}
            size={18}
            aria-hidden
            className="mt-0.5 shrink-0 text-foreground-tertiary"
          />
          <Text variant="body" as="span" tone="emphasis" className="min-w-0">
            {item.text}
          </Text>
        </li>
      ))}
    </ul>
  );
}
