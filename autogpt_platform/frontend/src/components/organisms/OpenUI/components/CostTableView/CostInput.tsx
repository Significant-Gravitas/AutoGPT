import { useId } from "react";
import { Input } from "@/components/atoms/Input/Input";
import { useFieldValidation } from "../useFieldValidation";
import { costInputError } from "./helpers";

interface Props {
  label: string;
  kind: "quantity" | "unitPrice";
  value: string | number;
  disabled: boolean;
  onChange: (value: string) => void;
}

export function CostInput({ label, kind, value, disabled, onChange }: Props) {
  const id = useId();
  const { error, ...handlers } = useFieldValidation(
    disabled ? "" : costInputError(value, kind),
  );
  return (
    <div className="min-w-0">
      <Input
        {...handlers}
        id={id}
        type="text"
        inputMode="decimal"
        label={label}
        labelVariant="body-medium"
        value={value}
        disabled={disabled}
        required={!disabled}
        maxLength={100}
        onChange={(event) => onChange(event.target.value)}
        aria-invalid={!!error}
        aria-describedby={`${id}-error`}
        wrapperClassName="mb-0"
      />
      <p
        id={`${id}-error`}
        aria-live="polite"
        className="text-xs text-destructive"
      >
        {error}
      </p>
    </div>
  );
}
