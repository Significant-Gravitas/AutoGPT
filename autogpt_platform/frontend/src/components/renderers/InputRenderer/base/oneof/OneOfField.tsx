import {
  descriptionId,
  FieldProps,
  getTemplate,
  getUiOptions,
  getWidget,
} from "@rjsf/utils";
import { useEffect, useRef, useState } from "react";
import { AnyOfField } from "../anyof/AnyOfField";
import { cleanUpHandleId, getHandleId, updateUiOption } from "../../helpers";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { ANY_OF_FLAG } from "../../constants";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

function getDiscriminatorPropName(schema: any): string | undefined {
  if (!schema?.discriminator) return undefined;
  if (typeof schema.discriminator === "string") return schema.discriminator;
  return schema.discriminator.propertyName;
}

export function OneOfField(props: FieldProps) {
  const { schema } = props;

  const discriminatorProp = getDiscriminatorPropName(schema);
  if (!discriminatorProp) {
    return <AnyOfField {...props} />;
  }

  return (
    <DiscriminatedUnionField {...props} discriminatorProp={discriminatorProp} />
  );
}

interface DiscriminatedUnionFieldProps extends FieldProps {
  discriminatorProp: string;
}

function DiscriminatedUnionField({
  discriminatorProp,
  ...props
}: DiscriminatedUnionFieldProps) {
  const { schema, registry, formData, onChange, name } = props;
  const { fields, schemaUtils, formContext } = registry;
  const { SchemaField } = fields;
  const { nodeId } = formContext;

  const field_id = props.fieldPathId.$id;

  // Resolve variant schemas from $refs
  const variants = useRef(
    (schema.oneOf || []).map((opt: any) =>
      schemaUtils.retrieveSchema(opt, formData),
    ),
  );

  // Build dropdown options from variant titles and discriminator const values
  const enumOptions = variants.current.map((variant: any, index: number) => {
    const discValue = (variant.properties?.[discriminatorProp] as any)?.const;
    return {
      value: index,
      label: variant.title || discValue || `Option ${index + 1}`,
      discriminatorValue: discValue,
    };
  });

  // Determine initial selected index from formData
  function getInitialIndex() {
    const currentDisc = formData?.[discriminatorProp];
    if (currentDisc) {
      const idx = enumOptions.findIndex(
        (o) => o.discriminatorValue === currentDisc,
      );
      if (idx >= 0) return idx;
    }
    return 0;
  }

  const [selectedIndex, setSelectedIndex] = useState(getInitialIndex);

  // Generate handleId for sub-fields (same convention as AnyOfField)
  const uiOptions = getUiOptions(props.uiSchema, props.globalUiOptions);
  const handleId = getHandleId({
    uiOptions,
    id: field_id + ANY_OF_FLAG,
    schema,
  });

  const childUiSchema = updateUiOption(props.uiSchema, {
    handleId,
    label: false,
    fromAnyOf: true,
  });

  // Get selected variant schema with discriminator property filtered out
  // and sub-fields inheriting the parent's advanced value
  const selectedVariant = variants.current[selectedIndex];
  const parentAdvanced = (schema as any).advanced;

  function getFilteredSchema() {
    if (!selectedVariant?.properties) return selectedVariant;
    const filteredProperties: Record<string, any> = {};
    for (const [key, value] of Object.entries(selectedVariant.properties)) {
      if (key === discriminatorProp) continue;
      filteredProperties[key] =
        parentAdvanced !== undefined
          ? { ...(value as any), advanced: parentAdvanced }
          : value;
    }
    return {
      ...selectedVariant,
      properties: filteredProperties,
      required: (selectedVariant.required || []).filter(
        (r: string) => r !== discriminatorProp,
      ),
    };
  }

  const filteredSchema = getFilteredSchema();

  // Handle variant change
  function handleVariantChange(option?: string) {
    const newIndex = option !== undefined ? parseInt(option, 10) : -1;
    if (newIndex === selectedIndex || newIndex < 0) return;

    const newVariant = variants.current[newIndex];
    const oldVariant = variants.current[selectedIndex];
    const discValue = (newVariant.properties?.[discriminatorProp] as any)
      ?.const;

    // Clean edges for this field
    const handlePrefix = cleanUpHandleId(field_id);
    useEdgeStore.getState().removeEdgesByHandlePrefix(nodeId, handlePrefix);

    // Sanitize current data against old→new schema to preserve shared fields
    let newFormData = schemaUtils.sanitizeDataForNewSchema(
      newVariant,
      oldVariant,
      formData,
    );

    // Fill in defaults for the new variant
    newFormData = schemaUtils.getDefaultFormState(
      newVariant,
      newFormData,
      "excludeObjectChildren",
    ) as any;
    newFormData = { ...newFormData, [discriminatorProp]: discValue };

    setSelectedIndex(newIndex);
    onChange(newFormData, props.fieldPathId.path, undefined, field_id);
  }

  // Sync selectedIndex when formData discriminator changes externally
  // (e.g. undo/redo, loading saved state)
  const currentDiscValue = formData?.[discriminatorProp];
  useEffect(() => {
    const idx = currentDiscValue
      ? enumOptions.findIndex((o) => o.discriminatorValue === currentDiscValue)
      : -1;

    if (idx >= 0) {
      if (idx !== selectedIndex) setSelectedIndex(idx);
    } else if (enumOptions.length > 0 && selectedIndex !== 0) {
      // Unknown or cleared discriminator — full reset via same cleanup path
      handleVariantChange("0");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentDiscValue]);

  // Auto-set discriminator on initial render if missing
  useEffect(() => {
    const discValue = enumOptions[selectedIndex]?.discriminatorValue;
    if (discValue && formData?.[discriminatorProp] !== discValue) {
      onChange(
        { ...formData, [discriminatorProp]: discValue },
        props.fieldPathId.path,
        undefined,
        field_id,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const Widget = getWidget({ type: "string" }, "select", registry.widgets);

  const selector = (
    <Widget
      id={field_id}
      name={`${name}__oneof_select`}
      schema={{ type: "number", default: 0 }}
      onChange={handleVariantChange}
      onBlur={props.onBlur}
      onFocus={props.onFocus}
      disabled={props.disabled || enumOptions.length === 0}
      multiple={false}
      value={selectedIndex}
      options={{ enumOptions }}
      registry={registry}
      placeholder={props.placeholder}
      autocomplete={props.autocomplete}
      className={cn("-ml-1 h-[22px] w-fit gap-1 px-1 pl-2 text-xs font-medium")}
      autofocus={props.autofocus}
      label=""
      hideLabel={true}
      readonly={props.readonly}
    />
  );

  const DescriptionFieldTemplate = getTemplate(
    "DescriptionFieldTemplate",
    registry,
    uiOptions,
  );
  const description_id = descriptionId(props.fieldPathId ?? "");

  return (
    <div>
      <div className="flex items-center gap-2">
        <Text variant="body" className="line-clamp-1">
          {schema.title || name}
        </Text>
        <Text variant="small" className="mr-1 text-red-500">
          {props.required ? "*" : null}
        </Text>
        {selector}
        <DescriptionFieldTemplate
          id={description_id}
          description={schema.description || ""}
          schema={schema}
          registry={registry}
        />
      </div>
      {filteredSchema && filteredSchema.type !== "null" && (
        <SchemaField
          {...props}
          schema={filteredSchema}
          uiSchema={childUiSchema}
        />
      )}
    </div>
  );
}
