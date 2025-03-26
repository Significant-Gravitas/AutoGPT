import React, { FC } from "react";
import { Input } from "@/components/ui/input";
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
  onChange: (e: { target: { value: any } }) => void;
}

/**
 * A generic, data-type-based input component that uses Shadcn UI.
 * It inspects the schema via `determineDataType` and renders
 * the correct UI component.
 */
export const TypeBasedInput: FC<TypeBasedInputProps> = ({
  schema,
  value,
  onChange,
}) => {
  // Determine which UI to show based on the schema
  const dataType = determineDataType(schema);

  // A small utility for changing the parent:
  // we unify the "value" in an event-like object.
  const handleValueChange = (newValue: any) => {
    onChange(newValue);
  };

  switch (dataType) {
    case DataType.NUMBER:
      // Render a numeric input
      return (
        <Input
          type="number"
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );

    case DataType.LONG_TEXT:
      // Render a multi-line text area
      return (
        <Textarea
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );

    case DataType.TOGGLE:
      // Render a boolean switch
      return (
        <Switch
          checked={!!value}
          onCheckedChange={(checked) => handleValueChange(checked)}
        />
      );

    case DataType.DATE:
      // Basic date input (HTML5). For a more advanced calendar, you could
      // use Shadcn's `<Popover>` + `<Calendar>` approach.
      return (
        <Input
          type="date"
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );

    case DataType.TIME:
      return (
        <Input
          type="time"
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );

    case DataType.DATE_TIME:
      // Render HTML5 datetime-local input.
      // Or a custom calendar/time pick solution from Shadcn.
      return (
        <Input
          type="datetime-local"
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );

    case DataType.FILE:
      // A simple file input that calls handleValueChange with the File object(s)
      return (
        <Input
          type="file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            handleValueChange(file || null);
          }}
        />
      );

    case DataType.SELECT:
      // If there's an enum present, show a dropdown.
      // This is a single-select example using Shadcn’s Select
      console.log(schema);
      if ("enum" in schema && Array.isArray(schema.enum)) {
        return (
          <Select
            value={value ?? ""}
            onValueChange={(val) => handleValueChange(val)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select an option" />
            </SelectTrigger>
            <SelectContent>
              {schema.enum.map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {String(opt)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      }
      // fallback if no `schema.enum`:
      return (
        <Input
          type="text"
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );

    case DataType.SHORT_TEXT:
    default:
      // Basic text input for short text, or fallback
      return (
        <Input
          type="text"
          value={value ?? ""}
          onChange={(e) => handleValueChange(e.target.value)}
        />
      );
  }
};
