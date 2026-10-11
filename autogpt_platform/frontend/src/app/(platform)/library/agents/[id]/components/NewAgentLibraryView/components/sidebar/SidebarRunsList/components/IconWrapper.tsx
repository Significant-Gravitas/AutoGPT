import { cn } from "@/lib/utils";

type Props = {
  children: React.ReactNode;
  className: string;
  label?: string;
};

export function IconWrapper({ children, className, label }: Props) {
  return (
    <div
      role={label ? "img" : undefined}
      aria-label={label}
      className={cn(
        "flex h-5 w-5 items-center justify-center rounded-large border",
        className,
      )}
    >
      {children}
    </div>
  );
}
