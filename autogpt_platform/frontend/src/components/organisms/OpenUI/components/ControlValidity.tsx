import { useFieldValidation } from "./useFieldValidation";

interface Props {
  label: string;
  error: string;
}

export function ControlValidity({ label, error }: Props) {
  const { error: _, ...handlers } = useFieldValidation(error);
  return (
    <>
      <input
        {...handlers}
        className="sr-only"
        aria-label={`${label} validation`}
        tabIndex={-1}
        required
        onChange={() => {}}
        value={error ? "" : "valid"}
      />
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </>
  );
}
