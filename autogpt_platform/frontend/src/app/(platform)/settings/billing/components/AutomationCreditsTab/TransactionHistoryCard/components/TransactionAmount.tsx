import { Text } from "@/components/atoms/Text/Text";
import { formatAmount } from "../helpers";

type Props = {
  amount: number;
  variant?: "body-medium" | "small-medium" | "small";
};

export function TransactionAmount({ amount, variant = "body-medium" }: Props) {
  const isCredit = amount > 0;
  return (
    <span
      className={
        isCredit
          ? "whitespace-nowrap text-success-foreground tabular-nums"
          : "whitespace-nowrap tabular-nums"
      }
    >
      <Text variant={variant} as="span" tone={isCredit ? "inherit" : undefined}>
        {formatAmount(amount)}
      </Text>
    </span>
  );
}
