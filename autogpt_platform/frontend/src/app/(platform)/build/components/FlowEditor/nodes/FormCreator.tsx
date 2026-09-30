import { RJSFSchema } from "@rjsf/utils";
import type { IChangeEvent } from "@rjsf/core";
import React, { useContext, useMemo } from "react";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType } from "../../types";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { gateDiscriminatorOptions } from "@/components/renderers/InputRenderer/custom/CredentialField/gateDiscriminatorOptions";
import { credentialNotApplicable } from "@/components/renderers/InputRenderer/custom/CredentialField/helpers";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";

function isCredentialsProperty(schema: RJSFSchema, key: string): boolean {
  const property = schema.properties?.[key];
  return (
    typeof property === "object" &&
    property !== null &&
    "credentials_provider" in property
  );
}

interface FormCreatorProps {
  jsonSchema: RJSFSchema;
  nodeId: string;
  uiType: BlockUIType;
  /** When true the block is an MCP Tool with a selected tool. */
  isMCPWithTool?: boolean;
  showHandles?: boolean;
  className?: string;
}

export const FormCreator: React.FC<FormCreatorProps> = React.memo(
  ({
    jsonSchema,
    nodeId,
    uiType,
    isMCPWithTool = false,
    showHandles = true,
    className,
  }) => {
    const updateNodeData = useNodeStore((state) => state.updateNodeData);

    const getHardCodedValues = useNodeStore(
      (state) => state.getHardCodedValues,
    );

    const credentialsProviders = useContext(CredentialsProvidersContext);

    const isAgent = uiType === BlockUIType.AGENT;

    const handleChange = ({
      formData,
    }: IChangeEvent<Record<string, unknown>>) => {
      if (!formData) return;

      // RJSF seeds `const` provider/type into default form state, so an
      // untouched credential field arrives as {provider, type} with no id.
      // That half object must never reach input_default: graph activation
      // indexes creds_meta["id"] and would raise KeyError.
      //
      // Keyed off the schema, not the field name. MCP tool arguments are
      // declared by third-party servers and may legitimately be named
      // `*_credentials` while being ordinary values; a name test deleted them.
      for (const key of Object.keys(formData)) {
        if (!isCredentialsProperty(jsonSchema, key)) continue;
        const value = formData[key] as
          | { id?: unknown; provider?: unknown }
          | undefined;
        // Also drop a credential the current selection no longer needs, or a
        // connection chosen under one transport survives a switch to another
        // that uses none — invisible, because its row is hidden.
        const property = jsonSchema.properties?.[key];
        const stale =
          !jsonSchema.required?.includes(key) &&
          typeof property === "object" &&
          property !== null &&
          credentialNotApplicable(
            formData,
            property as unknown as BlockIOCredentialsSubSchema,
            typeof value?.provider === "string" ? value.provider : undefined,
          );
        if (!value?.id || stale) {
          delete formData[key];
        }
      }

      let updatedValues;
      if (isAgent) {
        updatedValues = {
          ...getHardCodedValues(nodeId),
          inputs: formData,
        };
      } else if (isMCPWithTool) {
        // Separate credentials from tool arguments — credentials are stored
        // at the top level of hardcodedValues, not inside tool_arguments.
        const { credentials, ...toolArgs } = formData;
        const selected = credentials as { id?: unknown } | undefined;
        updatedValues = {
          ...getHardCodedValues(nodeId),
          tool_arguments: toolArgs,
          ...(selected?.id ? { credentials } : {}),
        };
      } else {
        updatedValues = formData;
      }

      updateNodeData(nodeId, { hardcodedValues: updatedValues });
    };

    const hardcodedValues = getHardCodedValues(nodeId);

    // Memoized so the object identity is stable across renders. FormRenderer
    // memoizes its schema preprocessing on these props, so a fresh object each
    // render re-ran the whole RJSF schema pipeline for every node on the canvas.
    const initialValues = useMemo(() => {
      if (isAgent) return hardcodedValues.inputs ?? {};
      if (isMCPWithTool) {
        // Merge tool arguments with credentials for the form
        return {
          ...(hardcodedValues.tool_arguments ?? {}),
          ...(hardcodedValues.credentials?.id
            ? { credentials: hardcodedValues.credentials }
            : {}),
        };
      }
      return hardcodedValues;
    }, [isAgent, isMCPWithTool, hardcodedValues]);

    // Domain gating lives here rather than in the credential field renderer
    // because the discriminator is a sibling property: only a whole-schema
    // view can map a credential field's discriminator back to the enum it
    // narrows. Run dialogs gate through backend aggregation instead.
    const gatedSchema = useMemo(
      () =>
        gateDiscriminatorOptions(
          jsonSchema,
          credentialsProviders,
          initialValues,
        ),
      [jsonSchema, credentialsProviders, initialValues],
    );

    return (
      <div
        className={className}
        data-id={`form-creator-container-${nodeId}-node`}
      >
        <FormRenderer
          jsonSchema={gatedSchema}
          handleChange={handleChange}
          uiSchema={uiSchema}
          initialValues={initialValues}
          formContext={{
            nodeId: nodeId,
            uiType: uiType,
            showHandles: showHandles,
            size: "small",
          }}
        />
      </div>
    );
  },
);

FormCreator.displayName = "FormCreator";
