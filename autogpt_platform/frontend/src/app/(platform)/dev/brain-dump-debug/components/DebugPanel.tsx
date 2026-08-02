import { Text } from "@/components/atoms/Text/Text";
import type { Icon } from "@phosphor-icons/react";
import type { ReactNode } from "react";

interface Props {
  title: string;
  description?: string;
  icon: Icon;
  action?: ReactNode;
  children: ReactNode;
}

export function DebugPanel({
  title,
  description,
  icon: PanelIcon,
  action,
  children,
}: Props) {
  return (
    <section className="rounded-2xlarge border border-zinc-200 bg-white p-8">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <PanelIcon size={22} className="mt-0.5 shrink-0 text-zinc-500" />
          <div className="flex flex-col gap-1">
            <Text variant="h5">{title}</Text>
            {description ? (
              <Text variant="small" className="max-w-prose text-zinc-500">
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
    <div className="flex flex-col gap-1 rounded-large bg-zinc-50 px-4 py-3">
      <Text variant="label" className="text-zinc-500">
        {label}
      </Text>
      <Text variant="body" unmask={false} className="break-all font-mono">
        {value}
      </Text>
    </div>
  );
}

export function DebugNote({ children }: { children: ReactNode }) {
  return (
    <Text variant="small" className="max-w-prose text-zinc-500">
      {children}
    </Text>
  );
}
