"use client";

import { useTallyPopup } from "./useTallyPopup";

export function TallyPopupSimple() {
  // Load the Tally script and set up event listeners
  useTallyPopup();
  return null;
}

export default TallyPopupSimple;
