import { FieldProps, RJSFSchema } from "@rjsf/utils";
import { CredentialField } from "../credentials/CredentialField";

export interface CustomFieldDefinition {
  id: string;
  matcher: (schema: any) => boolean;
  component: (props: FieldProps<any, RJSFSchema, any>) => JSX.Element;
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
    component: CredentialField,
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

export function getCustomFieldComponents() {
  return CUSTOM_FIELDS.reduce(
    (acc, field) => {
      acc[field.id] = field.component;
      return acc;
    },
    {} as Record<string, any>,
  );
}
