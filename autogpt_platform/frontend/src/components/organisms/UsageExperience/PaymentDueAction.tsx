import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  url: string;
  label: string;
  size?: "small" | "large";
  layout?: "stacked" | "inline";
}

export function PaymentDueAction({
  url,
  label,
  size = "large",
  layout = "stacked",
}: Props) {
  const stacked = layout === "stacked";
  return (
    <div
      data-slot="payment-due-action"
      className={cn(
        stacked ? "space-y-2.5" : "flex flex-wrap items-center gap-x-4 gap-y-2",
      )}
    >
      <Button
        as="NextLink"
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        size={size}
        className={stacked ? "w-full" : undefined}
        rightIcon={
          <Icon
            icon={LinkSquare01Icon}
            size={size === "small" ? 14 : 16}
            aria-hidden="true"
          />
        }
      >
        {label}
      </Button>
      <Text
        variant="small"
        tone="muted"
        className={cn("!text-xs", stacked && "text-center")}
      >
        Opens a secure Stripe page. You can use a different card there.
      </Text>
    </div>
  );
}
