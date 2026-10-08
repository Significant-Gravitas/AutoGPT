import {
  useFormName,
  useGetFieldValue,
  useIsStreaming,
  useSetFieldValue,
  useStateField,
} from "@openuidev/react-lang";
import { useEffect } from "react";

export function useFieldView<T extends string | number>(
  name: string,
  defaultValue: T,
) {
  const formName = useFormName();
  const getFieldValue = useGetFieldValue();
  const setFieldValue = useSetFieldValue();
  const isStreaming = useIsStreaming();

  useEffect(() => {
    if (!isStreaming && getFieldValue(formName, name) === undefined) {
      setFieldValue(formName, "Field", name, defaultValue, false);
    }
  }, [defaultValue, formName, getFieldValue, isStreaming, name, setFieldValue]);

  return useStateField(name, defaultValue);
}
