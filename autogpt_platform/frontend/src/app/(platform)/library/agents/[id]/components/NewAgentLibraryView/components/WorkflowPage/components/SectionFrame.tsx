import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";

interface Props {
  title?: string | null;
  onRemove?: () => void;
  children: ReactNode;
}

export function SectionFrame({ title, onRemove, children }: Props) {
  return (
    <section className="flex min-w-0 flex-col gap-3 rounded-2xl bg-surface p-5 smooth-shadow-ring-sm">
      {title || onRemove ? (
        <div className="flex items-center justify-between gap-2">
          {title ? (
            <Text
              variant="small-medium"
              tone="secondary"
              transform="uppercase"
              tracking="caps"
              className="break-words"
            >
              {title}
            </Text>
          ) : (
            <span />
          )}
          {onRemove ? (
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={`Remove ${title ? `the ${title.toLowerCase()} section` : "this section"}`}
              onClick={onRemove}
            >
              <Icon icon={Cancel01Icon} size={14} />
            </Button>
          ) : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}
