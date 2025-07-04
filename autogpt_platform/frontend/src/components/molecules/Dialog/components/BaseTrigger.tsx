import React, { PropsWithChildren } from "react";

import { useDialogCtx } from "../useDialogCtx";

export function BaseTrigger({ children }: PropsWithChildren) {
  const ctx = useDialogCtx();

  return React.cloneElement(children as React.ReactElement, {
    onClick: ctx.handleOpen,
    className: "cursor-pointer",
  });
}
