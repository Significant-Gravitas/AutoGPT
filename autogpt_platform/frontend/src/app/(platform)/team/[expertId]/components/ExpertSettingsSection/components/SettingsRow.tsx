import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  title: string;
  description: string;
  control: ReactNode;
  children?: ReactNode;
}

export function SettingsRow({ title, description, control, children }: Props) {
  return (
    <div className="flex flex-col gap-3 px-4 py-3.5 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="flex min-w-0 flex-col gap-0.5">
        <Text variant="body-medium" tone="primary">
          {title}
        </Text>
        <Text variant="small" tone="muted">
          {description}
        </Text>
        {children}
      </div>
      <div className="flex w-full shrink-0 sm:w-auto sm:justify-end">
        {control}
      </div>
    </div>
  );
}
