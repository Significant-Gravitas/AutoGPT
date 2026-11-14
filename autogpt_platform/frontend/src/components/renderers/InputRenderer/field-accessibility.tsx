import { createContext, ReactNode, useContext } from "react";
import { ExtendedFormContextType } from "./types";

interface FieldAccessibility {
  fieldId: string;
  labelId?: string;
  descriptionId?: string;
  errorId: string;
}

const FieldAccessibilityContext = createContext<FieldAccessibility | null>(
  null,
);

interface Props {
  value: FieldAccessibility;
  children: ReactNode;
}

export function FieldAccessibilityProvider({ value, children }: Props) {
  return (
    <FieldAccessibilityContext.Provider value={value}>
      {children}
    </FieldAccessibilityContext.Provider>
  );
}

export function getFieldDomId(
  id: string,
  formContext?: ExtendedFormContextType,
) {
  return `${formContext?.domIdPrefix ?? ""}${id}`;
}

export function useFieldAccessibility(
  id: string,
  label: string | undefined,
  formContext?: ExtendedFormContextType,
  describedBy?: string,
) {
  const field = useFieldAccessibilityContext(id);
  const descriptions = [describedBy, field?.descriptionId, field?.errorId]
    .filter(Boolean)
    .join(" ")
    .split(/\s+/)
    .filter(Boolean);

  return {
    id: getFieldDomId(id, formContext),
    "aria-labelledby": field?.labelId,
    "aria-label": field?.labelId ? undefined : label || undefined,
    "aria-describedby": [...new Set(descriptions)].join(" ") || undefined,
  };
}

export function useFieldAccessibilityContext(id: string) {
  const context = useContext(FieldAccessibilityContext);
  return context?.fieldId === id ? context : null;
}
