"use client";

import { useState } from "react";
import { Profession } from "./helpers";

export interface SelectedExpert {
  profession: Profession;
  layoutID: string;
  trigger: HTMLButtonElement;
}

export function useExpertPreview() {
  const [selected, setSelected] = useState<SelectedExpert | null>(null);

  function close() {
    setSelected(null);
  }

  return {
    selected,
    select: setSelected,
    close,
  };
}
