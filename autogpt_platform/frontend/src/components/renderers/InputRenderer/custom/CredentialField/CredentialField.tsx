import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Switch } from "@/components/atoms/Switch/Switch";
import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api";
import { Text } from "@/components/atoms/Text/Text";
import { FieldProps, getUiOptions } from "@rjsf/utils";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { CredentialFieldTitle } from "./components/CredentialFieldTitle";
import { useCredentialAvailability } from "./useCredentialAvailability";

export const CredentialsField = (props: FieldProps) => {
  const { formData, onChange, schema, registry, fieldPathId, required } = props;

  const formContext = registry.formContext;
  const uiOptions = getUiOptions(props.uiSchema);
  const nodeId = formContext?.nodeId;

  // Get sibling inputs (hardcoded values) and credentials optional state from the node store
  // Note: We select the node data directly instead of using getter functions to avoid
  // creating new object references that would cause infinite re-render loops with useShallow
  const { node, setCredentialsOptional } = useNodeStore(
    useShallow((state) => ({
      node: nodeId ? state.nodes.find((n) => n.id === nodeId) : undefined,
      setCredentialsOptional: state.setCredentialsOptional,
    })),
  );

  const hardcodedValues = useMemo(
    () => node?.data?.hardcodedValues || {},
    [node?.data?.hardcodedValues],
  );
  const credentialsOptional = useMemo(() => {
    const value = node?.data?.metadata?.credentials_optional;
    return typeof value === "boolean" ? value : false;
  }, [node?.data?.metadata?.credentials_optional]);

  const handleChange = (newValue: any) => {
    onChange(newValue, fieldPathId?.path);
  };

  const handleSelectCredentials = (credentialsMeta?: CredentialsMetaInput) => {
    if (credentialsMeta) {
      handleChange({
        id: credentialsMeta.id,
        provider: credentialsMeta.provider,
        title: credentialsMeta.title,
        type: credentialsMeta.type,
      });
    } else {
      handleChange(undefined);
    }
  };

  // Convert formData to CredentialsMetaInput format
  const selectedCredentials: CredentialsMetaInput | undefined = useMemo(
    () =>
      formData?.id
        ? {
            id: formData.id,
            provider: formData.provider,
            title: formData.title,
            type: formData.type,
          }
        : undefined,
    [formData?.id, formData?.provider, formData?.title, formData?.type],
  );

  // The schema's `required` array is authoritative in both surfaces. In the
  // builder canvas the node-level toggle can additionally relax a required
  // field to optional, but it must never mark a schema-optional field
  // required — blocks declaring an optional credential (default=None) would
  // otherwise render an unfillable star.
  const schemaRequired = nodeId ? !credentialsOptional && required : required;

  const availability = useCredentialAvailability(
    schema as BlockIOCredentialsSubSchema,
    hardcodedValues,
  );
  const isUnavailable = availability === "unavailable";

  // CredentialsInput renders nothing when the provider is missing from the
  // providers map, which used to leave a bare title with no control under it.
  // An optional field then has nothing actionable in it at all, so drop the row.
  if (isUnavailable && !schemaRequired) {
    return null;
  }

  // A provider this user cannot connect is not actionable, so the required
  // marker only misleads: it demands something the UI gives no way to supply.
  // The schema still requires it and execution-time validation still enforces
  // that — this only suppresses the star on a field that cannot be filled.
  const isRequired = isUnavailable ? false : schemaRequired;

  return (
    <div className="flex flex-col gap-2">
      <CredentialFieldTitle
        fieldPathId={fieldPathId}
        registry={registry}
        uiOptions={uiOptions}
        schema={schema}
        required={isRequired}
      />
      {availability === "unavailable" && (
        <Text variant="small" className="text-zinc-500">
          Not available on your account.
        </Text>
      )}
      <CredentialsInput
        schema={schema as BlockIOCredentialsSubSchema}
        selectedCredentials={selectedCredentials}
        onSelectCredentials={handleSelectCredentials}
        siblingInputs={hardcodedValues}
        showTitle={false}
        readOnly={formContext?.readOnly}
        isOptional={!isRequired}
        className="w-full"
        variant="node"
      />

      {/* Optional credentials toggle - only show in builder canvas, not run dialogs */}
      {nodeId &&
        !formContext?.readOnly &&
        formContext?.showOptionalToggle !== false && (
          <div className="mt-1 flex items-center gap-2">
            <Switch
              id={`credentials-optional-${nodeId}`}
              checked={credentialsOptional}
              onCheckedChange={(checked) =>
                setCredentialsOptional(nodeId, checked)
              }
            />
            <label
              htmlFor={`credentials-optional-${nodeId}`}
              className="cursor-pointer text-xs text-gray-500"
            >
              Optional - skip block if not configured
            </label>
          </div>
        )}
    </div>
  );
};
