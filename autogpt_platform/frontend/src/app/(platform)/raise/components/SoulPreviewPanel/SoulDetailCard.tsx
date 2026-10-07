"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { textClassFor } from "../ColorStep/helpers";

type Props = {
  label: string;
  value: string;
  color: string | null;
};

// Each answer arrives as its own card sliding up under the identity card, so
// the stack grows downward and lifts the name as it does.
export function SoulDetailCard({ label, value, color }: Props) {
  return (
    <div className="w-full rounded-3xl border border-border bg-background px-6 py-4 shadow-lg duration-500 animate-in fade-in slide-in-from-bottom-6 fill-mode-both motion-reduce:animate-none">
      <Text
        variant="small"
        className={cn(
          "mb-1 font-semibold uppercase tracking-[0.12em]",
          textClassFor(color) ?? "text-muted-foreground",
        )}
      >
        {label}
      </Text>
      <Text
        variant="large"
        unmask={false}
        className="line-clamp-3 text-[15px] leading-relaxed text-foreground"
      >
        {value}
      </Text>
    </div>
  );
}
