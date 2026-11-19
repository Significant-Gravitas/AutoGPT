import { useLayoutEffect, useRef, useSyncExternalStore } from "react";
import type {
  FieldValues,
  Path,
  PathValue,
  UseFormReturn,
} from "react-hook-form";

function subscribeToNothing() {
  return () => {};
}

/**
 * For a react-hook-form form that is in the server-rendered HTML. Until React
 * hydrates it the form has no submit handler, so `isHydrated` is false until
 * then and the caller keeps its buttons disabled. Text typed into the inputs
 * before hydration is in the page but not in the form, and the next render
 * would replace it with the form's empty value, so the form takes it over as
 * soon as it is hydrated.
 */
export function useServerRenderedForm<T extends FieldValues>(
  form: Pick<UseFormReturn<T>, "getValues" | "setValue">,
) {
  const formRef = useRef<HTMLFormElement>(null);
  const isHydrated = useSyncExternalStore(
    subscribeToNothing,
    () => true,
    () => false,
  );

  useLayoutEffect(() => {
    const elements = Array.from(formRef.current?.elements ?? []);
    for (const element of elements) {
      if (
        !(element instanceof HTMLInputElement) ||
        !element.name ||
        element.value === element.defaultValue
      ) {
        continue;
      }
      const name = element.name as Path<T>;
      if (typeof form.getValues(name) !== "string") continue;
      form.setValue(name, element.value as PathValue<T, Path<T>>, {
        shouldDirty: true,
      });
    }
  }, [form]);

  return { formRef, isHydrated };
}
