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
      className={`flex justify-end pt-6 gap-4 ${className}`}
      data-testid={testId}
    >
      {children}
    </div>
  ) : (
    <div
      className={`flex justify-between w-full pt-6 items-end gap-4 ${className}`}
      data-testid={testId}
    >
      {children}
    </div>
  );
}
