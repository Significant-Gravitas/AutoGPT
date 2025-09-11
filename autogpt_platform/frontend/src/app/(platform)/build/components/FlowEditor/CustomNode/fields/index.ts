import { RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "./CredentialField";
import { AnyField } from "./AnyField";

export const fields: RegistryFieldsType = {
  //   AnyField: AnyField,
  UnsupportedField: AnyField,
  credentials: CredentialsField,
};
