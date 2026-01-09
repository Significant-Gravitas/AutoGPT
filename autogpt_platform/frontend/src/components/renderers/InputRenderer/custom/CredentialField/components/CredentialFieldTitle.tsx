import {
  getTemplate,
  UiSchema,
  Registry,
  RJSFSchema,
  FieldPathId,
  titleId,
  descriptionId,
} from "@rjsf/utils";
import { getCredentialProviderFromSchema, toDisplayName } from "../helpers";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { updateUiOption } from "../../../helpers";
import { uiSchema } from "@/app/(platform)/build/components/FlowEditor/nodes/uiSchema";

export const CredentialFieldTitle = (props: {
  registry: Registry;
  uiOptions: UiSchema;
  schema: RJSFSchema;
  fieldPathId: FieldPathId;
  required?: boolean;
}) => {
  const { registry, uiOptions, schema, fieldPathId, required = false } = props;
  const { nodeId } = registry.formContext;

  const TitleFieldTemplate = getTemplate(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );

  const DescriptionFieldTemplate = getTemplate(
    "DescriptionFieldTemplate",
    registry,
    uiOptions,
  );

  const credentialProvider = toDisplayName(
    getCredentialProviderFromSchema(
      useNodeStore.getState().getHardCodedValues(nodeId),
      schema as BlockIOCredentialsSubSchema,
    ) ?? "",
  );

  const updatedUiSchema = updateUiOption(uiSchema, {
    showHandles: false,
  });

  return (
    <div className="flex items-center gap-2">
      <TitleFieldTemplate
        id={titleId(fieldPathId ?? "")}
        title={credentialProvider ?? ""}
        required={required}
        schema={schema}
        registry={registry}
        uiSchema={updatedUiSchema}
      />
      <DescriptionFieldTemplate
        id={descriptionId(fieldPathId ?? "")}
        description={schema.description || ""}
        schema={schema}
        registry={registry}
      />
    </div>
  );
};
