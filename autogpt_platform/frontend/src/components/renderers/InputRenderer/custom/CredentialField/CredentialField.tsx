import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Switch } from "@/components/atoms/Switch/Switch";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import type { CredentialField } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api";
import { FieldProps, getUiOptions } from "@rjsf/utils";
import { useCallback, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { CredentialFieldTitle } from "./components/CredentialFieldTitle";

const CREDENTIAL_KEY = "credentials";

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

  // In builder canvas (nodeId exists): show star based on credentialsOptional toggle
  // In run dialogs (no nodeId): show star based on schema's required array
  const isRequired = nodeId ? !credentialsOptional : required;

  // Convert single schema to CredentialField[] for CredentialsGroupedView
  const credentialFields: CredentialField[] = useMemo(
    () => [[CREDENTIAL_KEY, schema as BlockIOCredentialsSubSchema]],
    [schema],
  );

  const requiredCredentials = useMemo(
    () => (isRequired ? new Set([CREDENTIAL_KEY]) : new Set<string>()),
    [isRequired],
  );

  // Convert formData to inputCredentials map for CredentialsGroupedView
  const inputCredentials = useMemo(
    () => ({
      [CREDENTIAL_KEY]: formData?.id
        ? {
            id: formData.id,
            provider: formData.provider,
            title: formData.title,
            type: formData.type,
          }
        : undefined,
    }),
    [formData?.id, formData?.provider, formData?.title, formData?.type],
  );

  const handleCredentialChange = useCallback(
    (_key: string, value?: CredentialsMetaInput) => {
      if (value) {
        onChange(
          {
            id: value.id,
            provider: value.provider,
            title: value.title,
            type: value.type,
          },
          fieldPathId?.path,
        );
      } else {
        onChange(undefined, fieldPathId?.path);
      }
    },
    [onChange, fieldPathId?.path],
  );

  return (
    <div className="flex flex-col gap-2">
      <CredentialFieldTitle
        fieldPathId={fieldPathId}
        registry={registry}
        uiOptions={uiOptions}
        schema={schema}
        required={isRequired}
      />
      <CredentialsGroupedView
        credentialFields={credentialFields}
        requiredCredentials={requiredCredentials}
        inputCredentials={inputCredentials}
        inputValues={hardcodedValues}
        onCredentialChange={handleCredentialChange}
        showTitle={false}
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
