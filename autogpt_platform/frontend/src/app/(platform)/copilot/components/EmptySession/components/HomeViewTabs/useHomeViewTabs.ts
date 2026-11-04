import { useGlimm, type SweepHandle } from "glimm/react";
import { useRef, useState } from "react";
import type { HomeView } from "./useHomeView";

export function useHomeViewTabs(
  value: HomeView,
  onChange: (view: HomeView) => void,
) {
  const { sweep } = useGlimm();
  const [pending, setPending] = useState<HomeView | null>(null);
  const inFlight = useRef<SweepHandle | null>(null);

  function select(view: HomeView) {
    if (view === (pending ?? value)) return;
    inFlight.current?.cancel();
    setPending(view);
    const handle = sweep(() => onChange(view), {
      direction: view === "board" ? "ltr" : "rtl",
    });
    inFlight.current = handle;
    handle.done.finally(() => {
      if (inFlight.current !== handle) return;
      inFlight.current = null;
      setPending(null);
    });
  }

  return { selected: pending ?? value, select };
}
