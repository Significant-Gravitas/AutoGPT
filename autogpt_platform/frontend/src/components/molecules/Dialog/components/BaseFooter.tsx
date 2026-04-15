import { cn } from "@/lib/utils";
import { useDialogCtx } from "../useDialogCtx";

interface Props {
  children: React.ReactNode;
  testId?: string;
  className?: string;
  style?: React.CSSProperties;
}

export function BaseFooter({
  children,
  testId = "modal-footer",
  className,
  style,
}: Props) {
  const ctx = useDialogCtx();

  return ctx.isLargeScreen ? (
    <div
      className={cn("flex justify-end gap-4 pt-6", className)}
      data-testid={testId}
      style={style}
    >
      {children}
    </div>
  ) : (
    <div
      className={cn("flex w-full items-end justify-end gap-4 pt-6", className)}
      data-testid={testId}
    >
      {children}
    </div>
  );
}
