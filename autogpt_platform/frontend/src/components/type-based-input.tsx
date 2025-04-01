import React, { FC } from "react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import { CalendarIcon } from "lucide-react";
import { Cross2Icon, FileTextIcon } from "@radix-ui/react-icons";

import { Input as BaseInput } from "@/components/ui/input";
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

const inputClasses = "min-h-11 rounded-full border px-4 py-2.5";

function Input({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return <BaseInput {...props} className={cn(inputClasses, className)} />;
}

/**
 * A generic, data-type-based input component that uses Shadcn UI.
 * It inspects the schema via `determineDataType` and renders
 * the correct UI component.
 */
export const TypeBasedInput: FC<
  TypeBasedInputProps & React.HTMLAttributes<HTMLElement>
> = ({ schema, value, placeholder, onChange, ...props }) => {
  const dataType = determineDataType(schema);

  let innerInputElement: React.ReactNode = null;
  switch (dataType) {
    case DataType.NUMBER:
      innerInputElement = (
        <Input
          type="number"
          value={value ?? ""}
          placeholder={placeholder || "Enter number"}
          onChange={(e) => onChange(Number(e.target.value))}
          {...props}
        />
      );
      break;

    case DataType.LONG_TEXT:
      innerInputElement = (
        <Textarea
          className="rounded-[12px] px-3 py-2"
          value={value ?? ""}
          placeholder={placeholder || "Enter text"}
          onChange={(e) => onChange(e.target.value)}
          {...props}
        />
      );
      break;

    case DataType.BOOLEAN:
      innerInputElement = (
        <>
          <span className="text-sm text-gray-500">{placeholder}</span>
          <Switch
            className={placeholder ? "ml-auto" : "mx-auto"}
            checked={!!value}
            onCheckedChange={(checked) => onChange(checked)}
            {...props}
          />
        </>
      );
      break;

    case DataType.DATE:
      innerInputElement = (
        <DatePicker
          value={value}
          placeholder={placeholder}
          onChange={onChange}
          className={cn(inputClasses)}
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
        <Input
          type="datetime-local"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
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
          <Select value={value ?? ""} onValueChange={(val) => onChange(val)}>
            <SelectTrigger
              className={cn(inputClasses, "text-sm text-gray-500")}
            >
              <SelectValue placeholder={placeholder || "Select an option"} />
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
        break;
      }

    case DataType.SHORT_TEXT:
    default:
      innerInputElement = (
        <Input
          type="text"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder || "Enter text"}
          {...props}
        />
      );
  }

  return <div className="no-drag relative flex">{innerInputElement}</div>;
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

function getFileLabel(value: string) {
  if (value.startsWith("data:")) {
    const matches = value.match(/^data:([^;]+);/);
    if (matches?.[1]) {
      const mimeParts = matches[1].split("/");
      if (mimeParts.length > 1) {
        return `${mimeParts[1].toUpperCase()} file`;
      }
      return `${matches[1]} file`;
    }
  } else {
    const pathParts = value.split(".");
    if (pathParts.length > 1) {
      const ext = pathParts.pop();
      if (ext) return `${ext.toUpperCase()} file`;
    }
  }
  return "File";
}

function getFileSize(value: string) {
  if (value.startsWith("data:")) {
    const matches = value.match(/;base64,(.*)/);
    if (matches?.[1]) {
      const size = Math.ceil((matches[1].length * 3) / 4);
      if (size > 1024 * 1024) {
        return `${(size / (1024 * 1024)).toFixed(2)} MB`;
      } else {
        return `${(size / 1024).toFixed(2)} KB`;
      }
    }
  } else {
    return "";
  }
}

interface FileInputProps {
  value?: string; // base64 string or empty
  placeholder?: string; // e.g. "Resume", "Document", etc.
  onChange: (value: string) => void;
  className?: string;
}

const FileInput: FC<FileInputProps> = ({
  value,
  placeholder,
  onChange,
  className,
}) => {
  const loadFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64String = e.target?.result as string;
      onChange(base64String);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) loadFile(file);
  };

  const handleFileDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) loadFile(file);
  };

  const inputRef = React.useRef<HTMLInputElement>(null);

  return (
    <div className={cn("w-full", className)}>
      {value ? (
        <div className="flex min-h-14 items-center gap-4">
          <div className="agpt-border-input flex min-h-14 w-full items-center justify-between rounded-[12px] bg-zinc-50 p-4 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <FileTextIcon className="h-7 w-7 text-black" />
              <div className="flex flex-col gap-0.5">
                <span className="font-normal text-black">
                  {getFileLabel(value)}
                </span>
                <span>{getFileSize(value)}</span>
              </div>
            </div>
            <Cross2Icon
              className="h-5 w-5 cursor-pointer text-black"
              onClick={() => {
                inputRef.current && (inputRef.current.value = "");
                onChange("");
              }}
            />
          </div>
        </div>
      ) : (
        <div className="flex min-h-14 items-center gap-4">
          <div
            onDrop={handleFileDrop}
            onDragOver={(e) => e.preventDefault()}
            className="agpt-border-input flex min-h-14 w-full items-center justify-center rounded-[12px] border-dashed bg-zinc-50 text-sm text-gray-500"
          >
            Choose a file or drag and drop it here
          </div>

          <Button variant="default" onClick={() => inputRef.current?.click()}>
            Browse File
          </Button>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="*/*"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  );
};
