import { Input } from "@/components/atoms/Input/Input";
import { MAX_COLUMN_NAME } from "../../board-helpers";
import { useColumnNameInput } from "./useColumnNameInput";

interface Props {
  columnKey: string;
  name: string;
  onSave: (name: string) => void;
  onCancel: () => void;
  onDismiss: () => void;
}

export function ColumnNameInput({
  columnKey,
  name,
  onSave,
  onCancel,
  onDismiss,
}: Props) {
  const { draft, error, handleChange, handleKeyDown } = useColumnNameInput({
    name,
    onSave,
    onCancel,
  });

  return (
    <Input
      id={`column-name-${columnKey}`}
      label="Column name"
      hideLabel
      size="small"
      autoFocus
      autoComplete="off"
      maxLength={MAX_COLUMN_NAME}
      value={draft}
      error={error ?? undefined}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      onBlur={onDismiss}
      wrapperClassName="mb-0 min-w-0 flex-1"
    />
  );
}
