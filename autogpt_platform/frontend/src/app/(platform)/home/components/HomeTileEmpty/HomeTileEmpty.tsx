import type { IconSvgElement } from "@hugeicons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  icon: IconSvgElement;
  title: string;
  description: string;
  action?: { href: string; label: string };
  className?: string;
}

export function HomeTileEmpty({
  icon,
  title,
  description,
  action,
  className,
}: Props) {
  return (
    <div
      className={cn(
        "flex min-h-[14rem] flex-1 flex-col items-center justify-center gap-1 px-4 py-10 text-center",
        className,
      )}
    >
      <Icon
        icon={icon}
        size={22}
        className="text-zinc-400"
        aria-hidden="true"
      />
      <Text variant="body-medium" className="mt-1 text-zinc-800">
        {title}
      </Text>
      <Text variant="small" className="max-w-xs text-pretty text-zinc-500">
        {description}
      </Text>
      {action ? (
        <Button
          as="NextLink"
          href={action.href}
          variant="secondary"
          size="small"
          className="mt-3"
        >
          {action.label}
        </Button>
      ) : null}
    </div>
  );
}
