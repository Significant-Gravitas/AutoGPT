"use client";

import { Separator as KobraSeparator } from "@/components/ui/separator";

interface Props extends React.ComponentProps<typeof KobraSeparator> {
  /** Hidden from assistive tech (default). Base UI's separator is always semantic. */
  decorative?: boolean;
}

export function Separator({
  decorative = true,
  orientation = "horizontal",
  ...props
}: Props) {
  return (
    <KobraSeparator
      orientation={orientation}
      {...(decorative && { role: "none", "aria-orientation": undefined })}
      {...props}
    />
  );
}
