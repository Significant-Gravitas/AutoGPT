import { useEffect } from "react";
import {
  useFormName,
  useGetFieldValue,
  useIsStreaming,
  useSetFieldValue,
} from "@openuidev/react-lang";

export function useDerivedField(
  name: string,
  value: string | number | boolean | null,
) {
  const form = useFormName();
  const get = useGetFieldValue();
  const set = useSetFieldValue();
  const streaming = useIsStreaming();
  useEffect(() => {
    if (!streaming && !Object.is(get(form, name), value))
      set(form, "Derived", name, value);
  }, [form, get, name, set, streaming, value]);
}
