export function isMacPlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Mac|iPhone|iPad|iPod/.test(navigator.platform);
}

export function isEditableElement(el: Element | null): boolean {
  return (
    el instanceof HTMLInputElement ||
    el instanceof HTMLTextAreaElement ||
    (el instanceof HTMLElement && el.isContentEditable)
  );
}

export const COARSE_POINTER_QUERY = "(pointer: coarse)";

/** True when the primary input is a finger (phones, tablets). */
export function isTouchDevice(): boolean {
  return (
    typeof window !== "undefined" &&
    !!window.matchMedia &&
    window.matchMedia(COARSE_POINTER_QUERY).matches
  );
}
