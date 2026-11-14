import { useLayoutEffect, useState } from "react";

export function useFitListToDialog<T extends HTMLElement>() {
  const [list, setList] = useState<T | null>(null);

  useLayoutEffect(() => {
    if (!list) return;
    const body = list
      .closest("[data-dialog-content]")
      ?.querySelector<HTMLElement>('[class*="overflow-y-auto"]');
    if (!body) return;

    function fit() {
      if (!list || !body) return;
      list.style.maxHeight = "";
      const overflow = body.scrollHeight - body.clientHeight;
      const next =
        overflow > 0 ? Math.max(96, list.offsetHeight - overflow) : undefined;
      list.style.maxHeight = next === undefined ? "" : `${next}px`;
    }
    fit();
    window.addEventListener("resize", fit);
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fit);
    observer?.observe(body);
    observer?.observe(list);
    return () => {
      window.removeEventListener("resize", fit);
      observer?.disconnect();
    };
  }, [list]);
  return { attachList: setList, list };
}
