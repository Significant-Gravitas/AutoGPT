import { FieldProps, getFirstMatchingOption, mergeSchemas } from "@rjsf/utils";
import { useRef, useState } from "react";
import validator from "@rjsf/validator-ajv8";
import { getDefaultTypeIndex } from "./helpers";
import { cleanUpHandleId } from "../../helpers";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

export const useAnyOfField = (props: FieldProps) => {
  const { registry, schema, options, onChange, formData } = props;
  const { schemaUtils } = registry;

  const getInitialOption = () => {
    if (formData !== undefined && formData !== null) {
      const option = getFirstMatchingOption(
        validator,
        formData,
        options,
        schema,
      );
      return option !== undefined ? option : getDefaultTypeIndex(options);
    }
    return getDefaultTypeIndex(options);
  };

  const [selectedOption, setSelectedOption] =
    useState<number>(getInitialOption());
  const retrievedOptions = useRef<any[]>(
    options.map((opt: any) => schemaUtils.retrieveSchema(opt, formData)),
  );

  const option =
    selectedOption >= 0
      ? retrievedOptions.current[selectedOption] || null
      : null;
  let optionSchema: any | undefined | null;

  // adding top level required to each option schema
  if (option) {
    const { required } = schema;
    optionSchema = required
      ? (mergeSchemas({ required }, option) as any)
      : option;
  }

  const field_id = props.fieldPathId.$id;

  const handleOptionChange = (option?: string) => {
    const intOption = option !== undefined ? parseInt(option, 10) : -1;
    if (intOption === selectedOption) return;

    const newOption =
      intOption >= 0 ? retrievedOptions.current[intOption] : undefined;
    const oldOption =
      selectedOption >= 0
        ? retrievedOptions.current[selectedOption]
        : undefined;

    //   When we change the option, we need to clean the form data
    let newFormData = schemaUtils.sanitizeDataForNewSchema(
      newOption,
      oldOption,
      formData,
    );

    const handlePrefix = cleanUpHandleId(field_id);
    console.log("handlePrefix", handlePrefix);
    useEdgeStore
      .getState()
      .removeEdgesByHandlePrefix(registry.formContext.nodeId, handlePrefix);

    // We have cleaned the form data, now we need to get the default form state of new selected option
    if (newOption) {
      newFormData = schemaUtils.getDefaultFormState(
        newOption,
        newFormData,
        "excludeObjectChildren",
      ) as any;
    }

    setSelectedOption(intOption);
    onChange(newFormData, props.fieldPathId.path, undefined, field_id);
  };

  const enumOptions = retrievedOptions.current.map((option, index) => ({
    value: index,
    label: option.type,
  }));

  return {
    handleOptionChange,
    enumOptions,
    selectedOption,
    optionSchema,
    field_id,
  };
};
