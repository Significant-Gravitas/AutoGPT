"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { useNamingMomentCard } from "./useNamingMomentCard";

interface Props {
  className?: string;
}

export function NamingMomentCard({ className }: Props) {
  const { isEligible, dismiss, startNaming } = useNamingMomentCard();

  if (!isEligible) return null;

  return (
    <section
      aria-label="Name your AI"
      className={cn(
        "mx-auto w-full max-w-[42rem] rounded-3xl bg-white p-5 text-left shadow-zinc-950 smooth-shadow-ring-sm",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-purple-50 text-purple-600">
          <Icon icon={SparklesIcon} size={18} />
        </span>
        <Text variant="body-medium" className="text-zinc-800">
          I&apos;ve been working with you for a while. I think it&apos;s time I
          had a name.
        </Text>
      </div>
      <div className="mt-4 flex flex-wrap justify-end gap-2">
        <Button variant="ghost" size="small" onClick={dismiss}>
          Not now
        </Button>
        <Button variant="primary" size="small" onClick={startNaming}>
          Give me a name
        </Button>
      </div>
    </section>
  );
}
