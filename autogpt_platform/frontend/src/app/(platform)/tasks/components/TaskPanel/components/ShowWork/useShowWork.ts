import { useState } from "react";

export function useShowWork() {
  const [open, setOpen] = useState(false);

  return {
    open,
    toggle: () => setOpen((v) => !v),
  };
}
