import { cn } from "@/lib/utils";

type Props = {
  children: React.ReactNode;
  className?: string;
};

export function RunDetailCard({ children, className }: Props) {
  return (
    <div
      className={cn(
        "min-h-20 rounded-xlarge border border-slate-50/70 bg-white p-6",
        className,
      )}
    >
      {children}
    </div>
  );
}
