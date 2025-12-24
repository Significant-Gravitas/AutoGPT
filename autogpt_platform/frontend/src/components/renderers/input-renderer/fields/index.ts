import { RegistryFieldsType } from "@rjsf/utils";
import { CredentialsField } from "./CredentialField/CredentialField";
import { AnyOfField } from "./AnyOfField/AnyOfField";
import { ObjectField } from "./ObjectField";
import { LlmModelField } from "./LlmModelField/LlmModelField";

export const fields: RegistryFieldsType = {
  AnyOfField: AnyOfField,
  credentials: CredentialsField,
  ObjectField: ObjectField,
  llmModel: LlmModelField,
};
