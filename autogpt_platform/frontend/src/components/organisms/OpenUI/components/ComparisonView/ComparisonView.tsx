import { useId, type CSSProperties } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Comparison } from "@/lib/openui/catalog-connected";
import { Button } from "@/components/atoms/Button/Button";
import { useOpenUIDisabled } from "../../interactionContext";
import { useFieldValidation } from "../useFieldValidation";
import { ComparisonCard } from "./ComparisonCard";
import { useComparisonView } from "./useComparisonView";
import { useComparisonSwipe } from "./useComparisonSwipe";

export function ComparisonView({
  props,
}: ComponentRenderProps<z.infer<typeof Comparison.props>>) {
  const id = useId();
  const disabled = useOpenUIDisabled();
  const { selected, choose, undo, canUndo } = useComparisonView(props);
  const { error, ...validation } = useFieldValidation(
    selected ? "" : "Choose one option to continue.",
  );
  const { offset, candidate, ...gestures } = useComparisonSwipe(
    disabled,
    (index) => {
      const option = props.options?.[index];
      if (option) choose(option.id);
    },
  );
  return (
    <fieldset
      disabled={disabled}
      className="min-w-0 space-y-3"
      aria-describedby={`${id}-hint ${id}-error`}
    >
      <legend className="text-sm font-semibold text-foreground">
        {props.label}
      </legend>
      <p id={`${id}-hint`} className="text-xs text-muted-foreground md:hidden">
        Swipe left for A · right for B. Or use the buttons.
      </p>
      <input
        {...validation}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        value={selected?.id ?? ""}
        onChange={(event) => choose(event.target.value)}
        onFocus={() => document.getElementById(`${id}-0`)?.focus()}
      />
      <div className="overflow-hidden py-1">
        <div
          {...gestures}
          role="group"
          aria-label={`${props.label} swipe comparison`}
          data-resting={offset === 0}
          className="grid touch-pan-y grid-cols-2 gap-2 motion-safe:translate-x-[var(--swipe-x)] motion-safe:rotate-[var(--swipe-rotate)] motion-safe:data-[resting=true]:transition-transform motion-safe:data-[resting=true]:duration-200 md:gap-3"
          style={
            {
              "--swipe-x": `${offset}px`,
              "--swipe-rotate": `${offset / 40}deg`,
            } as CSSProperties
          }
        >
          {props.options?.map((option, index) => (
            <ComparisonCard
              key={option.id}
              option={option}
              index={index}
              selected={selected?.id === option.id}
              preview={candidate === index}
              disabled={disabled}
              onChoose={() => choose(option.id)}
              buttonID={`${id}-${index}`}
            />
          ))}
        </div>
      </div>
      <div className="flex min-h-9 items-center justify-between gap-2">
        <p role="status" className="text-xs text-muted-foreground">
          {selected
            ? `${selected.title} selected. Continue below when ready.`
            : "Your choice stays here until you continue."}
        </p>
        {canUndo && (
          <Button
            type="button"
            variant="ghost"
            size="small"
            className="min-h-11"
            disabled={disabled}
            onClick={undo}
          >
            Undo choice
          </Button>
        )}
      </div>
      <p
        id={`${id}-error`}
        aria-live="polite"
        className="text-xs text-destructive"
      >
        {error}
      </p>
    </fieldset>
  );
}
