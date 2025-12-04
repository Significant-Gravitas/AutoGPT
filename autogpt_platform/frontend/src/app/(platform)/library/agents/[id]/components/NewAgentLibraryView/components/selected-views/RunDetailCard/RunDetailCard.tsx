import { cn } from "@/lib/utils";

type Props = {
  children: React.ReactNode;
  className?: string;
};

export function RunDetailCard({ children, className }: Props) {
  return (
    <div
      className={cn(
        "mx-4 min-h-20 rounded-medium border border-zinc-100 bg-white p-6",
        className,
      )}
    >
      {children}
    </div>
  );
}
