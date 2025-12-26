import { uiSchema } from "@/app/(platform)/build/components/FlowEditor/nodes/uiSchema";
import {
  FieldProps,
  FormContextType,
  getUiOptions,
  getWidget,
  mergeSchemas,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { useRef, useState } from "react";

export const useAnyOfField = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: FieldProps<T, S, F>,
) => {
  const { registry, schema, options, formData, onChange } = props;
  const { schemaUtils } = registry;

  const [selectedOption, setSelectedOption] = useState<number>(0);
  const retrievedOptions = useRef<any[]>(
    options.map((opt: S) => schemaUtils.retrieveSchema(opt, formData)),
  );

  const option =
    selectedOption >= 0
      ? retrievedOptions.current[selectedOption] || null
      : null;
  let optionSchema: S | undefined | null;

  // adding top level required to each option schema
  if (option) {
    const { required } = schema;
    optionSchema = required
      ? (mergeSchemas({ required }, option) as S)
      : option;
  }

  const field_id = props.fieldPathId.$id;

  const handleOptionChange = (option?: string) => {
    console.log("calling handleOptionChange");
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

    // We have cleaned the form data, now we need to get the default form state of new selected option
    if (newOption) {
      newFormData = schemaUtils.getDefaultFormState(
        newOption,
        newFormData,
        "excludeObjectChildren",
      ) as T;
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
