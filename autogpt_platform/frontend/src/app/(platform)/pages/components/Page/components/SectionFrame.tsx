import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  Cancel01Icon,
  MessageEdit01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";
import type { PageEditing } from "../Page";

interface Props {
  index: number;
  title: string | null;
  editing?: PageEditing;
  isFirst: boolean;
  isLast: boolean;
  tone?: "takeaway" | "warning" | null;
  children: ReactNode;
}

export function SectionFrame({
  index,
  title,
  editing,
  isFirst,
  isLast,
  tone,
  children,
}: Props) {
  const name = title ? `the ${title} section` : "this section";
  const controls = editing ? (
    <div className="flex shrink-0 gap-0.5">
      <Control
        icon={ArrowUp01Icon}
        label={`Move ${name} up`}
        disabled={isFirst}
        onClick={() => editing.onMove(index, -1)}
      />
      <Control
        icon={ArrowDown01Icon}
        label={`Move ${name} down`}
        disabled={isLast}
        onClick={() => editing.onMove(index, 1)}
      />
      <Control
        icon={MessageEdit01Icon}
        label={`Change ${name} with Otto`}
        onClick={() => editing.onChangeWithOtto(index, title)}
      />
      <Control
        icon={Cancel01Icon}
        label={`Remove ${name}`}
        onClick={() => editing.onRemove(index)}
      />
    </div>
  ) : null;

  return (
    <section
      className={cn(
        "flex min-w-0 flex-col gap-3 rounded-2xl p-5 smooth-shadow-ring-sm",
        tone === "warning"
          ? "bg-warning-surface"
          : tone === "takeaway"
            ? "bg-surface-muted"
            : "bg-surface",
      )}
    >
      {title || controls ? (
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
          {controls}
        </div>
      ) : null}
      {children}
    </section>
  );
}

interface ControlProps {
  icon: IconSvgElement;
  label: string;
  disabled?: boolean;
  onClick(): void;
}

function Control({ icon, label, disabled, onClick }: ControlProps) {
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      <Icon icon={icon} size={14} />
    </Button>
  );
}
