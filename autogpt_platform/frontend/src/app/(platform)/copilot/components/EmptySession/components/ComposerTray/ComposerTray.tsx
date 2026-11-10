import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  recipientPicker: ReactNode;
  connection: ReactNode;
}

const RECIPIENT_LINK =
  "[&_button]:ml-0 [&_button]:h-7 [&_button]:gap-1.5 [&_button]:rounded-lg [&_button]:border-0 [&_button]:bg-transparent [&_button]:px-1.5 [&_button]:text-body [&_button]:font-normal [&_button]:text-foreground-ink [&_button]:shadow-none [&_button:hover]:bg-surface-tray [&_button>*:first-child]:size-5";

const CONNECTION_LINK =
  "[&_button]:h-7 [&_button]:min-w-0 [&_button]:gap-1 [&_button]:rounded-lg [&_button]:px-1.5 [&_button]:text-body-sm [&_button]:font-normal [&_button]:text-foreground-text [&_button:hover]:bg-surface-tray [&_button_svg]:size-3.5";

// The composer card sits inset in the gray tray; the tray's own footer names
// who the message goes to and what it runs on.
export function ComposerTray({ children, recipientPicker, connection }: Props) {
  return (
    <div className="rounded-composer-tray bg-surface-tray p-1">
      <div className="[&>form]:rounded-none [&>form]:bg-transparent [&>form]:p-0">
        {children}
      </div>
      <div className="flex h-9 items-center justify-between gap-2 pr-2 pl-1.5">
        <div className={cn("flex min-w-0 items-center", RECIPIENT_LINK)}>
          {recipientPicker}
        </div>
        <div className={cn("flex shrink-0 items-center", CONNECTION_LINK)}>
          {connection}
        </div>
      </div>
    </div>
  );
}
