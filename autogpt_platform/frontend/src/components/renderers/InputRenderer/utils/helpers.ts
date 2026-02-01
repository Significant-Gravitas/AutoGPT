export const getFieldErrorKey = (fieldId: string): string => {
  const withoutRoot = fieldId.startsWith("root_") ? fieldId.slice(5) : fieldId;
  return withoutRoot;
};
