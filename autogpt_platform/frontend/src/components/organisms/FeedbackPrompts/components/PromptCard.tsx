import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";

interface Props {
  eyebrow?: string;
  title: string;
  avatar?: ReactNode;
  children?: ReactNode;
  onClose: () => void;
  className?: string;
}

export function PromptCard({
  eyebrow,
  title,
  avatar,
  children,
  onClose,
  className,
}: Props) {
  return (
    <section
      aria-label={title}
      className={cn(
        "w-[28rem] rounded-2xl border border-border-base bg-surface p-4 shadow-lg",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        {avatar}
        <div className="flex flex-1 flex-col gap-0.5">
          {eyebrow ? (
            <Text
              variant="small"
              className="!text-xs !text-foreground-tertiary"
            >
              {eyebrow}
            </Text>
          ) : null}
          <Text variant="body-medium" className="!text-foreground-strong">
            {title}
          </Text>
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Close"
          withTooltip={false}
          onClick={onClose}
          className="-mr-1 -mt-1 shrink-0"
        >
          <Icon icon={Cancel01Icon} size={14} />
        </Button>
      </div>
      {children ? <div className="mt-4">{children}</div> : null}
    </section>
  );
}
