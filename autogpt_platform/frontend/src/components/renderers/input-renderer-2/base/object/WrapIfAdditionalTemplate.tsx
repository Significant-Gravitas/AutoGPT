import {
  ADDITIONAL_PROPERTY_FLAG,
  buttonId,
  getTemplate,
  getUiOptions,
  titleId,
  WrapIfAdditionalTemplateProps,
} from "@rjsf/utils";

import { Input } from "@/components/atoms/Input/Input";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

export default function WrapIfAdditionalTemplate(
  props: WrapIfAdditionalTemplateProps,
) {
  const {
    classNames,
    style,
    children,
    disabled,
    id,
    label,
    onRemoveProperty,
    onKeyRenameBlur,
    readonly,
    required,
    schema,
    uiSchema,
    registry,
  } = props;
  const { templates, formContext } = registry;
  const uiOptions = getUiOptions(uiSchema);
  // Button templates are not overridden in the uiSchema
  const { RemoveButton } = templates.ButtonTemplates;
  const { isInputConnected } = useEdgeStore();

  const additional = ADDITIONAL_PROPERTY_FLAG in schema;
  const { nodeId } = formContext;
  const handleId = uiOptions.handleId;

  const TitleFieldTemplate = getTemplate(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );

  if (!additional) {
    return (
      <div className={classNames} style={style}>
        {children}
      </div>
    );
  }

  const keyId = `${id}-key`;
  const generateObjectPropertyTitleId = (id: string, label: string) => {
    return id.replace(`_${label}`, `_#_${label}`);
  };
  const title_id = generateObjectPropertyTitleId(id, label);

  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    if (e.target.value == "") {
      onRemoveProperty();
    } else {
      onKeyRenameBlur(e);
    }
  };

  const isHandleConnected = isInputConnected(nodeId, handleId);

  return (
    <>
      <div className={`mb-4 flex flex-col gap-1`} style={style}>
        <TitleFieldTemplate
          id={titleId(title_id)}
          title={`#${label}`}
          required={required}
          schema={schema}
          registry={registry}
          uiSchema={uiSchema}
        />
        {!isHandleConnected && (
          <div className="flex flex-1 items-center gap-2">
            <Input
              label={""}
              hideLabel={true}
              required={required}
              defaultValue={label}
              disabled={disabled || readonly}
              id={keyId}
              wrapperClassName="mb-2 w-30"
              name={keyId}
              onBlur={!readonly ? handleBlur : undefined}
              type="text"
              size="small"
            />
            <div className="mt-2"> {children}</div>
          </div>
        )}
        {!isHandleConnected && (
          <div className="-mt-4">
            <RemoveButton
              id={buttonId(id, "remove")}
              disabled={disabled || readonly}
              onClick={onRemoveProperty}
              uiSchema={uiSchema}
              registry={registry}
            />
          </div>
        )}
      </div>
    </>
  );
}
