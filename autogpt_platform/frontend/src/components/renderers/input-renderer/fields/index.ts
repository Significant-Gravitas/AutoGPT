import { RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "../../input-renderer-2/custom/CredentialField/CredentialField";
import { AnyOfField } from "./AnyOfField/AnyOfField";
import { ObjectField } from "./ObjectField";

export const fields: RegistryFieldsType = {
  AnyOfField: AnyOfField,
  credentials: CredentialsField,
  ObjectField: ObjectField,
};
