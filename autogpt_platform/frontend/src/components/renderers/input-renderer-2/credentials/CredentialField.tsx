import {
  FieldProps,
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";

export const CredentialField = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: FieldProps<T, S, F>,
) => {
  return <div>CredentialField</div>;
};
