import { Flag02Icon, Wallet01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  from: string;
  capLine: string | null;
}

export function ExpectedBack({ from, capLine }: Props) {
  return (
    <div className="flex w-full shrink-0 flex-col gap-1.5 sm:w-60">
      <Text variant="eyebrow">Expected back</Text>
      <ExpectedItem icon={Flag02Icon}>
        A report in {from}&apos;s chat
      </ExpectedItem>
      {capLine ? (
        <ExpectedItem icon={Wallet01Icon}>{capLine}</ExpectedItem>
      ) : null}
    </div>
  );
}

function ExpectedItem({
  icon,
  children,
}: {
  icon: IconSvgElement;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon icon={icon} size={16} className="shrink-0 text-zinc-900" />
      <Text variant="body" tone="primary">
        {children}
      </Text>
    </div>
  );
}
