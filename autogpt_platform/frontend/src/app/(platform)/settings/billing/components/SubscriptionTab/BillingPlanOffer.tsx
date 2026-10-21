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
          ? "flex h-full flex-col rounded-2xl border border-zinc-200 bg-white p-6"
          : "flex h-full flex-col rounded-2xl border border-purple-200 bg-gradient-to-br from-purple-50 via-white to-blue-50 p-6"
      }
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 items-center justify-center rounded-lg bg-purple-100 text-purple-700">
          <Icon icon={SparklesIcon} size={17} />
        </span>
        <Text variant="small-medium" className="text-purple-800">
          AutoGPT {target}
        </Text>
      </div>
      <Text
        variant="h4"
        as="h2"
        className="mb-3 mt-4 text-[23px] leading-8 tracking-[-0.04em]"
      >
        {title}
      </Text>
      {description && (
        <Text variant="small" tone="secondary" className="mb-4 leading-6">
          {description}
        </Text>
      )}
      {price && (
        <div className="flex flex-wrap items-baseline gap-1.5">
          <Text
            variant="h2"
            as="span"
            className="font-sans text-[36px] font-semibold leading-10 tracking-[-0.045em]"
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
          <li key={benefit} className="flex items-start gap-2">
            <Icon
              icon={Tick02Icon}
              size={15}
              className="mt-0.5 shrink-0 text-purple-700"
            />
            <Text variant="small" className="text-zinc-700">
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
        variant="small"
        tone="secondary"
        className="mt-2.5 text-center text-xs leading-5"
      >
        {caption}
      </Text>
    </section>
  );
}
