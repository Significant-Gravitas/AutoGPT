import { cn } from "@/lib/utils";

type Props = {
  children: React.ReactNode;
  className: string;
};

export function IconWrapper({ children, className }: Props) {
  return (
    <div
      className={cn(
        "flex h-5 w-5 items-center justify-center rounded-large border",
        className,
      )}
    >
      {children}
    </div>
  );
}
