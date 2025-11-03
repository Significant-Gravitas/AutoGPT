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
  className = "",
  style,
}: Props) {
  const ctx = useDialogCtx();

  return ctx.isLargeScreen ? (
    <div
      className={`sticky bottom-0 mt-auto flex justify-end gap-4 border-t border-neutral-200 bg-white pt-6 dark:border-neutral-800 dark:bg-neutral-950 ${className}`}
      data-testid={testId}
      style={style}
    >
      {children}
    </div>
  ) : (
    <div
      className={`sticky bottom-0 mt-auto flex w-full items-end justify-between gap-4 border-t border-neutral-200 bg-white pt-6 dark:border-neutral-800 dark:bg-neutral-950 ${className}`}
      data-testid={testId}
    >
      {children}
    </div>
  );
}
