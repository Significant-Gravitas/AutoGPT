import { Text, type TextProps } from "@/components/atoms/Text/Text";
import { formatAmount } from "../helpers";

type Props = {
  amount: number;
  variant?: "body" | "small";
  weight?: TextProps["weight"];
};

export function TransactionAmount({
  amount,
  variant = "body",
  weight = "medium",
}: Props) {
  const isCredit = amount > 0;
  return (
    <span
      className={
        isCredit
          ? "whitespace-nowrap text-success-foreground tabular-nums"
          : "whitespace-nowrap tabular-nums"
      }
    >
      <Text
        variant={variant}
        weight={weight}
        as="span"
        tone={isCredit ? "inherit" : undefined}
      >
        {formatAmount(amount)}
      </Text>
    </span>
  );
}
