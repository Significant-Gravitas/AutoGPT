import {
  Alert as KobraAlert,
  AlertDescription as KobraAlertDescription,
  AlertTitle as KobraAlertTitle,
  type AlertTone,
} from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import * as React from "react";

type AlertVariant = AlertTone | "default";

interface Props extends React.ComponentProps<"div"> {
  children: React.ReactNode;
  /** `default` is the neutral notice; Kobra has no neutral tone, so it reads as `info`. */
  variant?: AlertVariant | null;
  /** Kobra draws the tone's own mark; a custom icon is not shown. */
  icon?: React.ComponentType<{ className?: string }>;
}

function Alert({ variant, icon: _icon, ...props }: Props) {
  const tone: AlertTone = !variant || variant === "default" ? "info" : variant;
  return <KobraAlert variant={tone} {...props} />;
}

// Kobra colours the text with the tone, which fails contrast on the tone's
// own tint (warning 1.8:1, success 3.3:1); the tone stays on the mark only.
function AlertTitle({
  className,
  ...props
}: React.ComponentProps<typeof KobraAlertTitle>) {
  return (
    <KobraAlertTitle className={cn("text-foreground", className)} {...props} />
  );
}

function AlertDescription({
  className,
  ...props
}: React.ComponentProps<typeof KobraAlertDescription>) {
  return (
    <KobraAlertDescription
      className={cn("text-foreground", className)}
      {...props}
    />
  );
}

export { Alert, AlertDescription, AlertTitle };
