import { useState, type MouseEvent } from "react";
import { asksBeforeOpening, isMiddlePress } from "../../helpers";

export function useExternalLinkButton(url: string) {
  const [isAsking, setIsAsking] = useState(false);

  // Bound to both click and auxclick (the middle button).
  function handleClick(event: MouseEvent<HTMLButtonElement>) {
    if (!asksBeforeOpening(event)) return;
    event.preventDefault();
    setIsAsking(true);
  }

  function handleMouseDown(event: MouseEvent<HTMLButtonElement>) {
    if (isMiddlePress(event)) event.preventDefault();
  }

  function handleConfirm() {
    window.open(url, "_blank", "noreferrer");
    setIsAsking(false);
  }

  function handleClose() {
    setIsAsking(false);
  }

  return { isAsking, handleClick, handleMouseDown, handleConfirm, handleClose };
}
