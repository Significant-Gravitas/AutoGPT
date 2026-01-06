import React, { useMemo } from "react";
import { FieldProps, getUiOptions } from "@rjsf/utils";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { CredentialFieldTitle } from "./components/CredentialFieldTitle";

export const CredentialsField = (props: FieldProps) => {
  const { formData, onChange, schema, registry, fieldPathId } = props;

  const formContext = registry.formContext;
  const uiOptions = getUiOptions(props.uiSchema);
  const nodeId = formContext?.nodeId;

  // Get sibling inputs (hardcoded values) from the node store
  const hardcodedValues = useNodeStore(
    useShallow((state) => (nodeId ? state.getHardCodedValues(nodeId) : {})),
  );

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

  return (
    <div className="flex flex-col gap-2">
      <CredentialFieldTitle
        fieldPathId={fieldPathId}
        registry={registry}
        uiOptions={uiOptions}
        schema={schema}
      />
      <CredentialsInput
        schema={schema as BlockIOCredentialsSubSchema}
        selectedCredentials={selectedCredentials}
        onSelectCredentials={handleSelectCredentials}
        siblingInputs={hardcodedValues}
        showTitle={false}
        readOnly={formContext?.readOnly}
      />
    </div>
  );
};
