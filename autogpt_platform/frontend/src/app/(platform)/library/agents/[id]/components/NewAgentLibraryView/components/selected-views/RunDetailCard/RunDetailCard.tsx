import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type Props = {
  children: React.ReactNode;
  className?: string;
  title?: React.ReactNode;
};

export function RunDetailCard({ children, className, title }: Props) {
  return (
    <div
      className={cn(
        "relative mx-4 flex min-h-20 flex-col gap-4 rounded-medium border border-zinc-100 bg-white p-6",
        className,
      )}
    >
      {title ? (
        typeof title === "string" ? (
          <Text variant="lead-semibold">{title}</Text>
        ) : (
          title
        )
      ) : null}
      {children}
    </div>
  );
}
