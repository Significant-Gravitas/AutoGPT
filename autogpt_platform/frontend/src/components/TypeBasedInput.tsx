import React, { FC } from "react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import { CalendarIcon } from "lucide-react";
import { Input as DefaultInput } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
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
  placeholder?: string;
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
  placeholder,
  onChange,
}) => (
  <div className="no-drag relative flex">
    {_TypeBasedInput({ schema, value, onChange, placeholder })}
  </div>
);

const _TypeBasedInput: FC<TypeBasedInputProps> = ({
  schema,
  value,
  placeholder,
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
          placeholder={placeholder}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      );

    case DataType.LONG_TEXT:
      return (
        <Textarea
          className="rounded-[12px] px-3 py-2"
          value={value ?? ""}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      );

    case DataType.TOGGLE: {
      return (
        <>
          <span className="text-sm text-gray-500">{placeholder}</span>
          <Switch
            className={placeholder ? "ml-auto" : "mx-auto"}
            checked={!!value}
            onCheckedChange={(checked) => onChange(checked)}
          />
        </>
      );
    }

    case DataType.DATE:
      return (
        <DatePicker
          value={value}
          placeholder={placeholder}
          onChange={onChange}
          className={cn(inputClasses, "max-w-xs")}
        />
      );

    case DataType.TIME:
      return <TimePicker value={value?.toString()} onChange={onChange} />;

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
            <SelectTrigger
              className={cn(inputClasses, "max-w-xs text-sm text-gray-500")}
            >
              <SelectValue placeholder="Select an option" />
            </SelectTrigger>
            <SelectContent className="rounded-[12px] border">
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

interface DatePickerProps {
  value?: Date;
  placeholder?: string;
  onChange: (date: Date | undefined) => void;
  className?: string;
}

export function DatePicker({
  value,
  placeholder,
  onChange,
  className,
}: DatePickerProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start font-normal",
            !value && "text-muted-foreground",
            className,
          )}
        >
          <CalendarIcon className="mr-2 h-5 w-5" />
          {value ? (
            format(value, "PPP")
          ) : (
            <span>{placeholder || "Pick a date"}</span>
          )}
        </Button>
      </PopoverTrigger>

      <PopoverContent className="flex min-h-[340px] w-auto p-0">
        <Calendar
          mode="single"
          selected={value}
          onSelect={(selected) => onChange(selected)}
          autoFocus
        />
      </PopoverContent>
    </Popover>
  );
}

interface TimePickerProps {
  value?: string;
  onChange: (time: string) => void;
  className?: string;
}

export function TimePicker({ value, onChange }: TimePickerProps) {
  const pad = (n: number) => n.toString().padStart(2, "0");
  const [hourNum, minuteNum] = value ? value.split(":").map(Number) : [0, 0];

  console.log(">>> hourNum", hourNum, "minuteNum", minuteNum, " value", value);

  const meridiem = hourNum >= 12 ? "PM" : "AM";
  const hour = pad(hourNum % 12 || 12);
  const minute = pad(minuteNum);

  const changeTime = (hour: string, minute: string, meridiem: string) => {
    const hour24 = (Number(hour) % 12) + (meridiem === "PM" ? 12 : 0);
    onChange(`${pad(hour24)}:${minute}`);
  };

  return (
    <div className="flex items-center space-x-3">
      <div className="flex flex-col items-center">
        <Select
          value={hour}
          onValueChange={(val) => changeTime(val, minute, meridiem)}
        >
          <SelectTrigger className={cn("text-center", inputClasses)}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Array.from({ length: 12 }, (_, i) => pad(i + 1)).map((h) => (
              <SelectItem key={h} value={h}>
                {h}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col items-center">
        <span className="m-auto text-xl font-bold">:</span>
      </div>

      <div className="flex flex-col items-center">
        <Select
          value={minute}
          onValueChange={(val) => changeTime(hour, val, meridiem)}
        >
          <SelectTrigger className={cn("text-center", inputClasses)}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Array.from({ length: 60 }, (_, i) => pad(i)).map((m) => (
              <SelectItem key={m} value={m.toString()}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col items-center">
        <Select
          value={meridiem}
          onValueChange={(val) => changeTime(hour, minute, val)}
        >
          <SelectTrigger className={cn("text-center", inputClasses)}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="AM">AM</SelectItem>
            <SelectItem value="PM">PM</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
