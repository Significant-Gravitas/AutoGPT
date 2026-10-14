import { useLayoutEffect, useRef } from "react";

// When the panel's button goes away (the apps are disconnected), the focus it
// held drops to the page, and the dialog's focus trap would park it at the top
// of the dialog. A layout effect catches it first, so the next Tab carries on
// from the panel.
export function useAlternativePanel(hasAction: boolean) {
  const panelRef = useRef<HTMLDivElement>(null);
  const hadAction = useRef(hasAction);

  useLayoutEffect(() => {
    const panel = panelRef.current;
    if (hadAction.current && !hasAction && panel && isFocusLost(panel)) {
      panel.focus();
    }
    hadAction.current = hasAction;
  }, [hasAction]);

  return { panelRef };
}

function isFocusLost(panel: HTMLElement) {
  const active = document.activeElement;
  return (
    !active ||
    active === document.body ||
    active === panel.closest('[role="dialog"]')
  );
}
