"use client";

import { useEffect, useState } from "react";
import { Profession } from "./helpers";

export interface SelectedExpert {
  profession: Profession;
  layoutId: string;
}

export function useExpertPreview() {
  const [selected, setSelected] = useState<SelectedExpert | null>(null);

  useEffect(() => {
    if (!selected) return;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setSelected(null);
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [selected]);

  return {
    selected,
    select: setSelected,
    close: () => setSelected(null),
  };
}
