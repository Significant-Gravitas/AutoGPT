import type { ReactNode } from "react";

interface Props {
  id: string;
  error: string;
  children: ReactNode;
}

export function FieldFeedback({ id, error, children }: Props) {
  return (
    <div>
      {children}
      <p id={id} aria-live="polite" className="-mt-5 text-xs text-red-600">
        {error}
      </p>
    </div>
  );
}
