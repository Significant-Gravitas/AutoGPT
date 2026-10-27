import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatAmount } from "../helpers";

type Props = { amount: number; className?: string };

export function TransactionAmount({ amount, className }: Props) {
  return (
    <Text
      variant="body-medium"
      as="span"
      className={cn(
        "whitespace-nowrap tabular-nums",
        amount > 0 ? "text-success-foreground" : "text-textBlack",
        className,
      )}
    >
      {formatAmount(amount)}
    </Text>
  );
}
