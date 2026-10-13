import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { ResultList } from "@/lib/openui/catalog-results";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { useOpenUIDisabled } from "../../interactionContext";
import { ControlValidity } from "../ControlValidity";
import { useResultList } from "./useResultList";
import { ResultCard } from "./ResultCard";

export function ResultListView({
  props,
}: ComponentRenderProps<z.infer<typeof ResultList.props>>) {
  const id = useId();
  const list = useResultList(props);
  const disabled = useOpenUIDisabled();
  return (
    <fieldset disabled={disabled} className="min-w-0 space-y-3">
      <legend className="text-sm font-semibold">{props.label}</legend>
      <Input
        id={id}
        label={`Search ${props.label}`}
        value={list.query}
        onChange={(event) => list.setQuery(event.target.value)}
      />
      <p className="text-xs text-muted-foreground" aria-live="polite">
        {list.matches.length} options · {list.selected.length} selected
        {props.unit ? ` · ${props.unit}` : ""}
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {list.visible.map((option) => (
          <ResultCard
            key={option.id}
            option={option}
            selected={list.selected.includes(option.id)}
            disabled={disabled}
            onSelect={() => list.choose(option.id)}
          />
        ))}
      </div>
      {!list.visible.length && (
        <p className="text-sm">
          No matching options. Your selections are kept.
        </p>
      )}
      <ControlValidity
        label={props.label}
        error={
          props.required && !list.selected.length
            ? "Choose an option before continuing."
            : ""
        }
      />
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="ghost"
          size="small"
          disabled={!list.page}
          onClick={() => list.setPage(list.page - 1)}
        >
          Previous options
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="small"
          disabled={(list.page + 1) * 6 >= list.matches.length}
          onClick={() => list.setPage(list.page + 1)}
        >
          More options
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="small"
          disabled={!list.canUndo}
          onClick={list.undo}
        >
          Undo selection
        </Button>
      </div>
    </fieldset>
  );
}
