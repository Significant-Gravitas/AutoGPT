import { useEffect, useState } from "react";

const BOTTOM_THRESHOLD_PX = 2;

export function useBottomScrollShadow(element: HTMLElement | null) {
  const [hasMoreBelow, setHasMoreBelow] = useState(false);

  useEffect(() => {
    if (!element) {
      setHasMoreBelow(false);
      return;
    }
    function update() {
      if (!element) return;
      const remaining =
        element.scrollHeight - element.scrollTop - element.clientHeight;
      setHasMoreBelow(remaining > BOTTOM_THRESHOLD_PX);
    }
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    function observeContents() {
      if (!element) return;
      observer?.disconnect();
      observer?.observe(element);
      Array.from(element.children).forEach((child) => observer?.observe(child));
      update();
    }
    observeContents();
    element.addEventListener("scroll", update, { passive: true });
    const mutations = new MutationObserver(observeContents);
    mutations.observe(element, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return () => {
      element.removeEventListener("scroll", update);
      observer?.disconnect();
      mutations.disconnect();
    };
  }, [element]);

  return element !== null && hasMoreBelow;
}
