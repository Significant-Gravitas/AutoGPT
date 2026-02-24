import { useState, useEffect, useCallback } from "react";
import { FieldProps } from "@rjsf/utils";
import { stringifyFormData, parseJsonValue, isValidJson } from "./helpers";

type FieldOnChange = FieldProps["onChange"];
type FieldPathId = FieldProps["fieldPathId"];

interface UseJsonTextFieldOptions {
  formData: unknown;
  onChange: FieldOnChange;
  path?: FieldPathId["path"];
}

interface UseJsonTextFieldReturn {
  textValue: string;
  isModalOpen: boolean;
  hasError: boolean;
  handleChange: (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => void;
  handleModalOpen: () => void;
  handleModalClose: () => void;
  handleModalSave: (value: string) => void;
}

/**
 * Custom hook for managing JSON text field state and handlers
 */
export function useJsonTextField({
  formData,
  onChange,
  path,
}: UseJsonTextFieldOptions): UseJsonTextFieldReturn {
  const [textValue, setTextValue] = useState(() => stringifyFormData(formData));
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [hasError, setHasError] = useState(false);

  // Update text value when formData changes externally
  useEffect(() => {
    const newValue = stringifyFormData(formData);
    setTextValue(newValue);
    setHasError(false);
  }, [formData]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const value = e.target.value;
      setTextValue(value);

      // Validate JSON and update error state
      const valid = isValidJson(value);
      setHasError(!valid);

      // Try to parse and update formData
      if (value.trim() === "") {
        onChange(undefined, path ?? []);
        return;
      }

      const parsed = parseJsonValue(value);
      if (parsed !== undefined) {
        onChange(parsed, path ?? []);
      }
    },
    [onChange, path],
  );

  const handleModalOpen = useCallback(() => {
    setIsModalOpen(true);
  }, []);

  const handleModalClose = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  const handleModalSave = useCallback(
    (value: string) => {
      setTextValue(value);
      setIsModalOpen(false);

      // Validate and update
      const valid = isValidJson(value);
      setHasError(!valid);

      if (value.trim() === "") {
        onChange(undefined, path ?? []);
        return;
      }

      const parsed = parseJsonValue(value);
      if (parsed !== undefined) {
        onChange(parsed, path ?? []);
      }
    },
    [onChange, path],
  );

  return {
    textValue,
    isModalOpen,
    hasError,
    handleChange,
    handleModalOpen,
    handleModalClose,
    handleModalSave,
  };
}
