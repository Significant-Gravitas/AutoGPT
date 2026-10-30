import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";

interface Props {
  title: string;
  description?: string;
  icon: IconSvgElement;
  action?: ReactNode;
  children: ReactNode;
}

export function DebugPanel({
  title,
  description,
  icon,
  action,
  children,
}: Props) {
  return (
    <section className="rounded-2xlarge border border-border-base bg-surface p-8">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <Icon
            icon={icon}
            size={22}
            className="mt-0.5 shrink-0 text-foreground-tertiary"
          />
          <div className="flex flex-col gap-1">
            <Text variant="h5">{title}</Text>
            {description ? (
              <Text variant="small" tone="muted" className="max-w-prose">
                {description}
              </Text>
            ) : null}
          </div>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

interface FieldProps {
  label: string;
  value: string;
}

export function DebugField({ label, value }: FieldProps) {
  return (
    <div className="flex flex-col gap-1 rounded-large bg-surface-muted px-4 py-3">
      <Text variant="label" tone="muted">
        {label}
      </Text>
      <Text variant="body" className="font-mono break-all">
        {value}
      </Text>
    </div>
  );
}

export function DebugNote({ children }: { children: ReactNode }) {
  return (
    <Text variant="small" tone="muted" className="max-w-prose">
      {children}
    </Text>
  );
}
