import { SparklesIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  target: string;
  title: string;
  quiet?: boolean;
  description?: string;
  price: string | null;
  benefits: string[];
  actionLabel: string;
  caption: string;
  onAction?: () => void;
  disabled?: boolean;
}

export function BillingPlanOffer({
  target,
  title,
  quiet = false,
  description,
  price,
  benefits,
  actionLabel,
  caption,
  onAction,
  disabled,
}: Props) {
  return (
    <section
      aria-label={`AutoGPT ${target}`}
      className={
        quiet
          ? "flex h-full flex-col rounded-2xl border border-border-base bg-surface p-6"
          : "flex h-full flex-col rounded-2xl border border-brand/30 bg-gradient-to-br from-accent-surface via-surface to-info-surface p-6"
      }
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 items-center justify-center rounded-lg bg-accent-surface text-accent-strong">
          <Icon icon={SparklesIcon} size={17} />
        </span>
        <Text
          variant="small-medium"
          // oxlint-disable-next-line design-system/no-restyle -- no Text tone for accent-strong yet (Text helpers belong to P2)
          className="text-accent-strong"
        >
          AutoGPT {target}
        </Text>
      </div>
      <Text
        variant="h4"
        as="h2"
        // oxlint-disable-next-line design-system/no-restyle -- the plan title is set tight on a 32px line; no Text variant has it
        className="mt-4 mb-3 leading-8 tracking-display"
      >
        {title}
      </Text>
      {description && (
        // oxlint-disable-next-line design-system/no-restyle -- caption copy at a 24px line height; no Text variant has it
        <Text variant="small" tone="secondary" className="mb-4 leading-6">
          {description}
        </Text>
      )}
      {price && (
        <div className="flex flex-wrap items-baseline gap-1.5">
          <Text
            variant="h2"
            as="span"
            // oxlint-disable-next-line design-system/no-restyle -- the price is a sans display number; no Text variant has it
            className="font-sans text-title leading-10 font-semibold tracking-display"
          >
            {price.split(" / ")[0]}
          </Text>
          <Text as="span" variant="small" tone="secondary">
            / {price.split(" / ")[1]}
          </Text>
        </div>
      )}
      <ul className="my-5 space-y-2.5">
        {benefits.map((benefit) => (
          <li
            key={benefit}
            className="flex items-start gap-2 text-accent-strong"
          >
            <Icon icon={Tick02Icon} size={15} className="mt-0.5 shrink-0" />
            <Text variant="small" tone="body">
              {benefit}
            </Text>
          </li>
        ))}
      </ul>
      <Button
        variant={quiet ? "secondary" : "primary"}
        size="large"
        className="mt-auto w-full"
        onClick={onAction}
        disabled={disabled}
      >
        {actionLabel}
      </Button>
      <Text
        variant="small-loose"
        tone="secondary"
        className="mt-2.5 text-center"
      >
        {caption}
      </Text>
    </section>
  );
}
