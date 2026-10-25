"use client";

import { Message01Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

export function SupportOffer({
  eyebrow,
  title,
  description,
  label,
  href,
  ready = false,
}: {
  eyebrow: string;
  title: string;
  description: string;
  label: string;
  href: string;
  ready?: boolean;
}) {
  return (
    <section
      aria-label={ready ? "Your plan is ready" : "Plan support"}
      className="flex h-full flex-col rounded-2xl border border-border-base bg-surface-muted/60 p-6"
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 items-center justify-center rounded-lg bg-surface text-foreground-secondary ring-1 ring-border-base">
          <Icon icon={ready ? SparklesIcon : Message01Icon} size={17} />
        </span>
        <Text variant="small-medium" className="text-foreground-secondary">
          {eyebrow}
        </Text>
      </div>
      <Text
        variant="h4"
        as="h2"
        // TODO(tokens): SECRT-3152 rare size (text-[23px])
        className="mb-3 mt-4 text-[23px] leading-8 tracking-display"
      >
        {title}
      </Text>
      <Text variant="small" tone="secondary" className="mb-7 leading-6">
        {description}
      </Text>
      <Button
        variant="secondary"
        size="large"
        className="mt-auto w-full"
        as="NextLink"
        href={href}
      >
        {label}
      </Button>
    </section>
  );
}
