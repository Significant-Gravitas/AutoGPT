import { useDialogCtx } from "../useDialogCtx";

interface Props {
  children: React.ReactNode;
  testId?: string;
  className?: string;
}

export function BaseFooter({
  children,
  testId = "modal-footer",
  className = "",
}: Props) {
  const ctx = useDialogCtx();

  return ctx.isLargeScreen ? (
    <div
      className={`flex justify-end gap-4 pt-6 ${className}`}
      data-testid={testId}
    >
      {children}
    </div>
  ) : (
    <div
      className={`flex w-full items-end justify-between gap-4 pt-6 ${className}`}
      data-testid={testId}
    >
      {children}
    </div>
  );
}
