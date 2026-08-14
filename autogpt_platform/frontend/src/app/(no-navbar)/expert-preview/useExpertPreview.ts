"use client";

import { useState } from "react";
import { Profession } from "./helpers";

export interface SelectedExpert {
  profession: Profession;
  layoutId: string;
  trigger: HTMLButtonElement;
}

export function useExpertPreview() {
  const [selected, setSelected] = useState<SelectedExpert | null>(null);

  return {
    selected,
    select: setSelected,
    close: () => setSelected(null),
  };
}
