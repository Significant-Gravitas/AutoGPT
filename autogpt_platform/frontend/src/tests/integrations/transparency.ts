/** True when the element or an ancestor is drawn at opacity 0, as an
 *  entrance animation's starting style leaves it until the animation runs. */
export function isDrawnTransparent(element: Element | null) {
  for (let node = element; node; node = node.parentElement) {
    if (node instanceof HTMLElement && node.style.opacity === "0") return true;
  }
  return false;
}
