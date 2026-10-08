"use client";

import { Switch as KobraSwitch } from "@/components/ui/switch";
import { forwardRef } from "react";

type Props = React.ComponentProps<typeof KobraSwitch>;

export const Switch = forwardRef<HTMLButtonElement, Props>(
  function Switch(props, ref) {
    return <KobraSwitch ref={ref} {...props} />;
  },
);
