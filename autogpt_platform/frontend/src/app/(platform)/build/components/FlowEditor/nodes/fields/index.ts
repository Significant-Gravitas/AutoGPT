import { RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "./CredentialField";
import { AnyOfField } from "./AnyOfField/AnyOfField";
import { ObjectField } from "./ObjectField";
import { NullField } from "./NullField";

export const fields: RegistryFieldsType = {
  AnyOfField: AnyOfField,
  credentials: CredentialsField,
  ObjectField: ObjectField,
  NullField: NullField,
};
