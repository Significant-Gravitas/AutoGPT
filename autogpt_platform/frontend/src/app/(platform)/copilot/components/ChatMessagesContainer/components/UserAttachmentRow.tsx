import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

/** A user message's attachments: above its bubble, right-aligned. */
export function UserAttachmentRow({ children }: Props) {
  return (
    <div
      data-testid="user-message-attachments"
      className="flex max-w-full flex-wrap justify-end gap-2"
    >
      {children}
    </div>
  );
}
