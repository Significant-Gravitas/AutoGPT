import React from "react";
import { format } from "date-fns";

import { Input as DSInput } from "@/components/atoms/Input/Input";
import { Select as DSSelect } from "@/components/atoms/Select/Select";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";
// Removed shadcn Select usage in favor of DS Select for time picker
import {
  BlockIOObjectSubSchema,
  BlockIOSubSchema,
  DataType,
  determineDataType,
} from "@/lib/autogpt-server-api/types";
import { TimePicker } from "@/components/molecules/TimePicker/TimePicker";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { useRunAgentInputs } from "./useRunAgentInputs";
import { Switch } from "@/components/atoms/Switch/Switch";

/**
 * A generic prop structure for the TypeBasedInput.
 *
 * onChange expects an event-like object with e.target.value so the parent
 * can do something like setInputValues(e.target.value).
 */
interface Props {
  schema: BlockIOSubSchema;
  value?: any;
  placeholder?: string;
  onChange: (value: any) => void;
}

/**
 * A generic, data-type-based input component that uses Shadcn UI.
 * It inspects the schema via `determineDataType` and renders
 * the correct UI component.
 */
export function RunAgentInputs({
  schema,
  value,
  placeholder,
  onChange,
  ...props
}: Props & React.HTMLAttributes<HTMLElement>) {
  const { handleUploadFile, uploadProgress } = useRunAgentInputs();

  const dataType = determineDataType(schema);
  const baseId = String(schema.title ?? "input")
    .replace(/\s+/g, "-")
    .toLowerCase();

  let innerInputElement: React.ReactNode = null;
  switch (dataType) {
    case DataType.NUMBER:
      innerInputElement = (
        <DSInput
          id={`${baseId}-number`}
          label={schema.title ?? placeholder ?? "Number"}
          hideLabel
          size="small"
          type="number"
          value={value ?? ""}
          placeholder={placeholder || "Enter number"}
          onChange={(e) =>
            onChange(Number((e.target as HTMLInputElement).value))
          }
          {...props}
        />
      );
      break;

    case DataType.LONG_TEXT:
      innerInputElement = (
        <DSInput
          id={`${baseId}-textarea`}
          label={schema.title ?? placeholder ?? "Text"}
          hideLabel
          size="small"
          type="textarea"
          rows={3}
          value={value ?? ""}
          placeholder={placeholder || "Enter text"}
          onChange={(e) => onChange((e.target as HTMLInputElement).value)}
          {...props}
        />
      );
      break;

    case DataType.BOOLEAN:
      innerInputElement = (
        <>
          <span className="text-sm text-gray-500">
            {placeholder || (value ? "Enabled" : "Disabled")}
          </span>
          <Switch
            className="ml-auto"
            checked={!!value}
            onCheckedChange={(checked: boolean) => onChange(checked)}
            {...props}
          />
        </>
      );
      break;

    case DataType.DATE:
      innerInputElement = (
        <DSInput
          id={`${baseId}-date`}
          label={schema.title ?? placeholder ?? "Date"}
          hideLabel
          size="small"
          type="date"
          value={value ? format(value as Date, "yyyy-MM-dd") : ""}
          onChange={(e) => {
            const v = (e.target as HTMLInputElement).value;
            if (!v) onChange(undefined);
            else {
              const [y, m, d] = v.split("-").map(Number);
              onChange(new Date(y, m - 1, d));
            }
          }}
          placeholder={placeholder || "Pick a date"}
          {...props}
        />
      );
      break;

    case DataType.TIME:
      innerInputElement = (
        <TimePicker value={value?.toString()} onChange={onChange} />
      );
      break;

    case DataType.DATE_TIME:
      innerInputElement = (
        <DSInput
          id={`${baseId}-datetime`}
          label={schema.title ?? placeholder ?? "Date time"}
          hideLabel
          size="small"
          type="datetime-local"
          value={value ?? ""}
          onChange={(e) => onChange((e.target as HTMLInputElement).value)}
          placeholder={placeholder || "Enter date and time"}
          {...props}
        />
      );
      break;

    case DataType.FILE:
      innerInputElement = (
        <FileInput
          value={value}
          placeholder={placeholder}
          onChange={onChange}
          onUploadFile={handleUploadFile}
          uploadProgress={uploadProgress}
          {...props}
        />
      );
      break;

    case DataType.SELECT:
      if (
        "enum" in schema &&
        Array.isArray(schema.enum) &&
        schema.enum.length > 0
      ) {
        innerInputElement = (
          <DSSelect
            id={`${baseId}-select`}
            label={schema.title ?? placeholder ?? "Select"}
            hideLabel
            value={value ?? ""}
            size="small"
            onValueChange={(val: string) => onChange(val)}
            placeholder={placeholder || "Select an option"}
            options={schema.enum
              .filter((opt) => opt)
              .map((opt) => ({ value: opt, label: String(opt) }))}
          />
        );
        break;
      }

    case DataType.MULTI_SELECT: {
      const _schema = schema as BlockIOObjectSubSchema;
      const allKeys = Object.keys(_schema.properties);
      const selectedValues = Object.entries(value || {})
        .filter(([_, v]) => v)
        .map(([k]) => k);

      innerInputElement = (
        <MultiToggle
          items={allKeys.map((key) => ({
            value: key,
            label: _schema.properties[key]?.title ?? key,
            size: "small",
          }))}
          selectedValues={selectedValues}
          onChange={(values: string[]) =>
            onChange(
              Object.fromEntries(
                allKeys.map((opt) => [opt, values.includes(opt)]),
              ),
            )
          }
          className="nodrag"
          aria-label={schema.title}
        />
      );
      break;
    }

    case DataType.SHORT_TEXT:
    default:
      innerInputElement = (
        <DSInput
          id={`${baseId}-text`}
          label={schema.title ?? placeholder ?? "Text"}
          hideLabel
          size="small"
          type="text"
          value={value ?? ""}
          onChange={(e) => onChange((e.target as HTMLInputElement).value)}
          placeholder={placeholder || "Enter text"}
          {...props}
        />
      );
  }

  return (
    <div className="no-drag relative flex w-full">{innerInputElement}</div>
  );
}
