import { RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "./CredentialField";
import { AnyOfField } from "./AnyOfField/AnyOfField";
import { ObjectField } from "./ObjectField";

export const fields: RegistryFieldsType = {
  AnyOfField: AnyOfField,
  credentials: CredentialsField,
  ObjectField: ObjectField,
};
