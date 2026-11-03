import { Button } from "@/components/atoms/Button/Button";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { KeyboardEvent, PointerEvent } from "react";
import { PANEL_MIN_WIDTH } from "../helpers";

interface Props {
  panelID: string;
  width: number;
  maxWidth?: number;
  onCollapse(): void;
  handlers: {
    onPointerDown(e: PointerEvent<HTMLElement>): void;
    onPointerMove(e: PointerEvent<HTMLElement>): void;
    onPointerUp(e: PointerEvent<HTMLElement>): void;
    onPointerCancel(e: PointerEvent<HTMLElement>): void;
    onKeyDown(e: KeyboardEvent<HTMLElement>): void;
  };
}

export function ResizeHandle({
  panelID,
  width,
  maxWidth,
  onCollapse,
  handlers,
}: Props) {
  return (
    <div className="group relative flex h-full w-5 shrink-0 justify-center">
      <button
        type="button"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize task details"
        aria-controls={panelID}
        aria-valuenow={width}
        aria-valuemin={PANEL_MIN_WIDTH}
        aria-valuemax={maxWidth}
        className="flex h-full w-full cursor-col-resize touch-none justify-center rounded-control focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        {...handlers}
      >
        <span className="h-full w-px bg-transparent transition-colors duration-150 ease-out-expo group-hover:bg-border-hairline" />
      </button>
      <div className="absolute top-3 opacity-0 transition-opacity duration-150 ease-out-expo group-hover:opacity-100 focus-within:opacity-100">
        <Button
          variant="icon"
          size="icon-2xs"
          aria-label="Hide task details"
          aria-expanded
          aria-controls={panelID}
          leadingIcon={ArrowRight01Icon}
          onClick={onCollapse}
        />
      </div>
    </div>
  );
}
