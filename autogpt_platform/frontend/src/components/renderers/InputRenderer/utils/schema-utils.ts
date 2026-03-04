import { getUiOptions, RJSFSchema, UiSchema } from "@rjsf/utils";

export function isAnyOfSchema(schema: RJSFSchema | undefined): boolean {
  return (
    Array.isArray(schema?.anyOf) &&
    schema!.anyOf.length > 0 &&
    schema?.enum === undefined
  );
}

export const isAnyOfChild = (
  uiSchema: UiSchema<any, RJSFSchema, any> | undefined,
): boolean => {
  const uiOptions = getUiOptions(uiSchema);
  return uiOptions.label === false;
};

export function isOptionalType(schema: RJSFSchema | undefined): {
  isOptional: boolean;
  type?: any;
} {
  if (
    !Array.isArray(schema?.anyOf) ||
    schema!.anyOf.length !== 2 ||
    !schema!.anyOf.some((opt: any) => opt.type === "null")
  ) {
    return { isOptional: false };
  }

  const nonNullType = schema!.anyOf?.find((opt: any) => opt.type !== "null");

  return {
    isOptional: true,
    type: nonNullType,
  };
}
export function isAnyOfSelector(name: string) {
  return name.includes("anyof_select");
}

export function isMultiSelectSchema(schema: RJSFSchema | undefined): boolean {
  if (typeof schema !== "object" || schema === null) {
    return false;
  }

  if ("anyOf" in schema || "oneOf" in schema) {
    return false;
  }

  return !!(
    schema.type === "object" &&
    schema.properties &&
    Object.values(schema.properties).every(
      (prop: any) => prop.type === "boolean",
    )
  );
}

const isGoogleDriveFileObject = (obj: RJSFSchema): boolean => {
  if (obj.type !== "object" || !obj.properties) {
    return false;
  }
  const props = obj.properties;
  const hasId = "id" in props;
  const hasMimeType = "mimeType" in props || "mime_type" in props;
  const hasIconUrl = "iconUrl" in props || "icon_url" in props;
  const hasIsFolder = "isFolder" in props || "is_folder" in props;
  return hasId && hasMimeType && (hasIconUrl || hasIsFolder);
};

export const isGoogleDrivePickerSchema = (
  schema: RJSFSchema | undefined,
): boolean => {
  if (!schema) {
    return false;
  }

  // highest priority
  if (
    "google_drive_picker_config" in schema ||
    ("format" in schema && schema.format === "google-drive-picker")
  ) {
    return true;
  }

  // In the Input type block, we do not add the format for the GoogleFile field, so we need to include this extra check.
  if (isGoogleDriveFileObject(schema)) {
    return true;
  }

  return false;
};
