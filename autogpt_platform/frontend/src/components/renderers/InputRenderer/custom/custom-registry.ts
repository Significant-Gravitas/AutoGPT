import { FieldProps, RJSFSchema, RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "./CredentialField/CredentialField";
import { GoogleDrivePickerField } from "./GoogleDrivePickerField/GoogleDrivePickerField";

export interface CustomFieldDefinition {
  id: string;
  matcher: (schema: any) => boolean;
  component: (props: FieldProps<any, RJSFSchema, any>) => JSX.Element | null;
}

export const CUSTOM_FIELDS: CustomFieldDefinition[] = [
  {
    id: "custom/credential_field",
    matcher: (schema: any) => {
      return (
        typeof schema === "object" &&
        schema !== null &&
        "credentials_provider" in schema
      );
    },
    component: CredentialsField,
  },
  {
    id: "custom/google_drive_picker_field",
    matcher: (schema: any) => {
      return (
        "google_drive_picker_config" in schema ||
        ("format" in schema && schema.format === "google-drive-picker")
      );
    },
    component: GoogleDrivePickerField,
  },
];

export function findCustomFieldId(schema: any): string | null {
  for (const field of CUSTOM_FIELDS) {
    if (field.matcher(schema)) {
      return field.id;
    }
  }
  return null;
}

export function generateCustomFields(): RegistryFieldsType {
  return CUSTOM_FIELDS.reduce(
    (acc, field) => {
      acc[field.id] = field.component;
      return acc;
    },
    {} as Record<string, any>,
  );
}
