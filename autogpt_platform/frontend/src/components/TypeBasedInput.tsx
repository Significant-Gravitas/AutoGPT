import React, { FC } from "react";
import { cn } from "@/lib/utils";
import { Input as DefaultInput } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { determineDataType, DataType } from "@/lib/autogpt-server-api/types";
import { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";

/**
 * A generic prop structure for the TypeBasedInput.
 *
 * onChange expects an event-like object with e.target.value so the parent
 * can do something like setInputValues(e.target.value).
 */
export interface TypeBasedInputProps {
  schema: BlockIOSubSchema;
  value?: any;
  onChange: (value: any) => void;
}

const inputClasses = "min-h-11 rounded-[55px] border px-4 py-2.5";

const Input = function Input({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return <DefaultInput {...props} className={cn(inputClasses, className)} />;
};

/**
 * A generic, data-type-based input component that uses Shadcn UI.
 * It inspects the schema via `determineDataType` and renders
 * the correct UI component.
 */
export const TypeBasedInput: FC<TypeBasedInputProps> = ({
  schema,
  value,
  onChange,
}) => (
  <div className="no-drag relative flex">
    {_TypeBasedInput({ schema, value, onChange })}
  </div>
);

const _TypeBasedInput: FC<TypeBasedInputProps> = ({
  schema,
  value,
  onChange,
}) => {
  // Determine which UI to show based on the schema
  const dataType = determineDataType(schema);

  switch (dataType) {
    case DataType.NUMBER:
      return (
        <Input
          type="number"
          value={value ?? ""}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      );

    case DataType.LONG_TEXT:
      return (
        <Textarea
          className="rounded-[12px] px-3 py-2"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case DataType.TOGGLE: {
      return (
        <Switch
          className="ml-auto"
          checked={!!value}
          onCheckedChange={(checked) => onChange(checked)}
        />
      );
    }

    case DataType.DATE:
      return (
        <Input
          type="date"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case DataType.TIME:
      return (
        <Input
          type="time"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case DataType.DATE_TIME:
      return (
        <Input
          type="datetime-local"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case DataType.FILE:
      return (
        <Input
          type="file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            onChange(file || null);
          }}
        />
      );

    case DataType.SELECT:
      if (
        "enum" in schema &&
        Array.isArray(schema.enum) &&
        schema.enum.length > 0
      ) {
        return (
          <Select value={value ?? ""} onValueChange={(val) => onChange(val)}>
            <SelectTrigger className={cn(inputClasses)}>
              <SelectValue placeholder="Select an option" />
            </SelectTrigger>
            <SelectContent>
              {schema.enum
                .filter((opt) => opt)
                .map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {String(opt)}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        );
      }

      return (
        <Input
          type="text"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case DataType.SHORT_TEXT:
    default:
      return (
        <Input
          type="text"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
};
