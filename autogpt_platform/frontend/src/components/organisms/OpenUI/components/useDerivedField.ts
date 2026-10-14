import { useEffect } from "react";
import {
  useFormName,
  useGetFieldValue,
  useIsStreaming,
  useSetFieldValue,
} from "@openuidev/react-lang";

export function useDerivedField(name: string, value: unknown) {
  const form = useFormName();
  const get = useGetFieldValue();
  const set = useSetFieldValue();
  const streaming = useIsStreaming();
  useEffect(() => {
    if (!streaming && JSON.stringify(get(form, name)) !== JSON.stringify(value))
      set(form, "Derived", name, value);
  }, [form, get, name, set, streaming, value]);
}
