import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  title: string;
  label: string;
  tone?: "default" | "danger";
  children: ReactNode;
}

export function SettingsGroup({
  title,
  label,
  tone = "default",
  children,
}: Props) {
  const isDanger = tone === "danger";

  return (
    <section aria-label={label} className="flex flex-col gap-3">
      <Text
        variant="body-medium"
        as="h3"
        tone={isDanger ? "danger" : "primary"}
      >
        {title}
      </Text>
      <div
        className={cn(
          "flex flex-col divide-y rounded-2xl",
          isDanger
            ? "divide-red-100 border border-red-200 bg-red-50/50"
            : "divide-zinc-100 bg-white smooth-shadow-ring-sm",
        )}
      >
        {children}
      </div>
    </section>
  );
}
