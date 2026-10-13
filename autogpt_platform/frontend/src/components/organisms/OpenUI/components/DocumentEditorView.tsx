import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { DocumentEditor } from "@/lib/openui/catalog-inputs";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { downloadText } from "@/lib/openui/download";
import { useFieldView } from "./useFieldView";
import { useOpenUIDisabled } from "../interactionContext";

export function DocumentEditorView({
  props,
}: ComponentRenderProps<z.infer<typeof DocumentEditor.props>>) {
  const id = useId();
  const field = useFieldView(props.name, props.value);
  const disabled = useOpenUIDisabled();
  const value = typeof field.value === "string" ? field.value : props.value;
  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-sm font-semibold">{props.title}</legend>
      <Input
        id={id}
        label="Current draft"
        type="textarea"
        rows={10}
        value={value}
        required={props.required}
        maxLength={20_000}
        onChange={(event) => field.setValue(event.target.value)}
      />
      <p className="text-xs text-muted-foreground" aria-live="polite">
        {value === props.original
          ? "Matches the original"
          : "Unsaved changes to review"}{" "}
        · {value.length} characters
      </p>
      <details>
        <summary className="min-h-11 cursor-pointer py-3 text-sm">
          Review original
        </summary>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted p-3 text-xs">
          {props.original}
        </pre>
      </details>
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="secondary"
          size="small"
          onClick={() =>
            downloadText(props.filename, value, "text/plain;charset=utf-8")
          }
        >
          Download current draft
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="small"
          disabled={value === props.original}
          onClick={() => field.setValue(props.original)}
        >
          Restore original
        </Button>
      </div>
    </fieldset>
  );
}
