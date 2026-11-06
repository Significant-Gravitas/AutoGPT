import { useState } from "react";
import {
  asNumber,
  ErrorSchema,
  FieldPathList,
  FieldProps,
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";

const INTEGER_LITERAL = /^[+-]?\d+$/;
// RJSF keeps a trailing decimal point or zeroes in the input while the user is
// typing, without storing them in the form data.
const TRAILING_ZERO_WITH_PREFIX = /\.(?:[0-9]*0)?$/;
const TRAILING_ZERO = /[0.]0*$/;

function isUnsafeIntegerLiteral(value: unknown): value is string {
  return (
    typeof value === "string" &&
    INTEGER_LITERAL.test(value.trim()) &&
    !Number.isSafeInteger(Number(value))
  );
}

/**
 * RJSF's `NumberField` runs every widget value through `asNumber`, which
 * rounds integers above `Number.MAX_SAFE_INTEGER` before the form data sees
 * them. Keep such literals as the typed string so an exact value survives the
 * round trip; every other value follows RJSF's own handling.
 */
export default function NumberField<
  T = unknown,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = Record<string, unknown>,
>(props: FieldProps<T, S, F>) {
  const { registry, onChange, formData, value: initialValue } = props;
  const [lastValue, setLastValue] = useState(initialValue);
  const { StringField } = registry.fields;

  let value = formData;

  function handleChange(
    next: FieldProps<T, S, F>["value"],
    path: FieldPathList,
    errorSchema?: ErrorSchema<T>,
    id?: string,
  ) {
    setLastValue(next);

    if (isUnsafeIntegerLiteral(next)) {
      onChange(next as T, path, errorSchema, id);
      return;
    }

    let candidate: unknown = next;
    if (`${candidate}`.charAt(0) === ".") {
      candidate = `0${candidate}`;
    }

    const processed =
      typeof candidate === "string" &&
      candidate.match(TRAILING_ZERO_WITH_PREFIX)
        ? asNumber(candidate.replace(TRAILING_ZERO, ""))
        : asNumber(candidate as string | null);

    onChange(processed as T, path, errorSchema, id);
  }

  if (typeof lastValue === "string" && typeof value === "number") {
    const re = new RegExp(`^(${String(value).replace(".", "\\.")})?\\.?0*$`);
    if (lastValue.match(re)) {
      value = lastValue as T;
    }
  }

  return <StringField {...props} formData={value} onChange={handleChange} />;
}
