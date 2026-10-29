import { Text } from "@/components/atoms/Text/Text";
import { formatAmount } from "../helpers";

type Props = {
  amount: number;
  variant?: "body-medium" | "small-medium" | "small";
};

export function TransactionAmount({ amount, variant = "body-medium" }: Props) {
  return (
    <Text
      variant={variant}
      as="span"
      /* oxlint-disable design-system/no-restyle -- amounts use tabular figures and credits read green; no Text tone for success-foreground yet */
      className={
        amount > 0
          ? "whitespace-nowrap tabular-nums text-success-foreground"
          : "whitespace-nowrap tabular-nums"
      }
      /* oxlint-enable design-system/no-restyle */
    >
      {formatAmount(amount)}
    </Text>
  );
}
