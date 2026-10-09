interface NumberConstraints {
  min: number | null;
  max: number | null;
  step: number;
  value: number;
}

export function numberFieldError(raw: string, props: NumberConstraints) {
  const text = raw.trim();
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text))
    return "Enter a number, such as 10 or 10.5.";
  const value = Number(text);
  if (!Number.isFinite(value)) return "Enter a finite number.";
  if (props.min !== null && value < props.min)
    return `Enter ${props.min} or more.`;
  if (props.max !== null && value > props.max)
    return `Enter ${props.max} or less.`;
  const base = props.min ?? props.value;
  const steps = (value - base) / props.step;
  if (Math.abs(steps - Math.round(steps)) > 1e-8)
    return `Use increments of ${props.step} starting at ${base}.`;
  return "";
}

export function validateForm(form: HTMLFormElement) {
  if (form.checkValidity()) return true;
  const fields = form.querySelectorAll<
    HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
  >("input, textarea, select");
  Array.from(fields)
    .find((field) => !field.validity.valid)
    ?.focus();
  return false;
}
