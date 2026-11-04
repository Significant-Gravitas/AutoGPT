import { useState, type MouseEvent } from "react";
import { asksBeforeOpening } from "./helpers";

export function useMessageLink(href: string | undefined) {
  const [isAsking, setIsAsking] = useState(false);

  // Bound to both click and auxclick (the middle button), so the one table in
  // helpers decides for every kind of click.
  function handleClick(event: MouseEvent<HTMLAnchorElement>) {
    if (!href || !asksBeforeOpening(event, href)) return;
    event.preventDefault();
    setIsAsking(true);
  }

  function handleConfirm() {
    if (href) window.open(href, "_blank", "noreferrer");
    setIsAsking(false);
  }

  function handleClose() {
    setIsAsking(false);
  }

  return { isAsking, handleClick, handleConfirm, handleClose };
}
