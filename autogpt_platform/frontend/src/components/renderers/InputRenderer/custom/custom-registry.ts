import { FieldProps, RJSFSchema, RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "./CredentialField/CredentialField";
import { GoogleDrivePickerField } from "./GoogleDrivePickerField/GoogleDrivePickerField";
import { JsonTextField } from "./JsonTextField/JsonTextField";
import { MultiSelectField } from "./MultiSelectField/MultiSelectField";
import {
  isGoogleDrivePickerSchema,
  isMultiSelectSchema,
} from "../utils/schema-utils";
import { TableField } from "./TableField/TableField";
import { LlmModelField } from "./LlmModelField/LlmModelField";

export interface CustomFieldDefinition {
  id: string;
  matcher: (schema: any) => boolean;
  component: (props: FieldProps<any, RJSFSchema, any>) => JSX.Element | null;
}

/** Field ID for JsonTextField - used to render nested complex types as text input */
export const JSON_TEXT_FIELD_ID = "custom/json_text_field";

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
    matcher: isGoogleDrivePickerSchema,
    component: GoogleDrivePickerField,
  },
  {
    id: "custom/json_text_field",
    // Not matched by schema - assigned via uiSchema for nested complex types
    matcher: () => false,
    component: JsonTextField,
  },
  {
    id: "custom/multi_select_field",
    matcher: isMultiSelectSchema,
    component: MultiSelectField,
  },
  {
    id: "custom/table_field",
    matcher: (schema: any) => {
      return (
        schema.type === "array" &&
        "format" in schema &&
        schema.format === "table"
      );
    },
    component: TableField,
  },
  {
    id: "custom/llm_model_field",
    matcher: (schema: any) => {
      return (
        typeof schema === "object" && schema !== null && "llm_model" in schema
      );
    },
    component: LlmModelField,
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
