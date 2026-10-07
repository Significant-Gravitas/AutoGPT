"use client";

import * as React from "react";
import {
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
} from "@radix-ui/react-icons";
import { DayPicker, DropdownProps } from "react-day-picker";

import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/__legacy__/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./select";

export type CalendarProps = React.ComponentProps<typeof DayPicker>;

function Calendar({
  className,
  classNames,
  showOutsideDays = true,
  ...props
}: CalendarProps) {
  return (
    <DayPicker
      showOutsideDays={showOutsideDays}
      captionLayout={"dropdown"}
      endMonth={new Date(2100, 0)}
      startMonth={new Date(1900, 0)}
      className={cn("p-3", className)}
      classNames={{
        months: "flex flex-col sm:flex-row space-y-4 sm:space-x-4 sm:space-y-0",
        month: "space-y-4",
        month_caption: "flex  items-center justify-center text-sm font-medium",
        dropdowns: "flex gap-2 ",
        caption_label: "hidden",
        nav: "hidden",
        button_previous:
          "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100",
        button_next: "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100",
        month_grid: "w-full border-collapse space-y-1",
        weekdays: "flex",
        weekday: "text-zinc-500 rounded-md w-8 font-normal text-[0.8rem]",
        week: "flex w-full mt-2",
        day: cn(
          "relative p-0 text-center text-sm focus-within:relative focus-within:z-20 has-aria-[selected]:bg-zinc-100 [&:has([aria-selected].day-outside)]:bg-zinc-100/50 [&:has([aria-selected].day-range-end)]:rounded-r-md",
          props.mode === "range"
            ? "[&:has(>.day-range-end)]:rounded-r-md [&:has(>.day-range-start)]:rounded-l-md first:has-aria-[selected]:rounded-l-md last:has-aria-[selected]:rounded-r-md"
            : "has-aria-[selected]:rounded-md",
        ),
        day_button: cn(
          buttonVariants({ variant: "ghost" }),
          "h-8 w-8 p-0 font-normal aria-selected:opacity-100",
        ),
        range_start: "range-start",
        range_end: "range-end",
        selected:
          "bg-zinc-900 text-zinc-100 hover:bg-zinc-900 hover:text-zinc-50 focus:bg-zinc-700 focus:text-zinc-50",
        today: "bg-zinc-100 text-zinc-900",
        outside:
          "day-outside text-zinc-500 opacity-50 aria-selected:bg-zinc-100/50 aria-selected:text-zinc-500 aria-selected:opacity-30",
        disabled: "text-zinc-500 opacity-50",
        range_middle: "aria-selected:bg-zinc-100 aria-selected:text-zinc-900",
        hidden: "invisible",
        ...classNames,
      }}
      components={{
        Chevron: (props) => {
          if (props.orientation === "left") {
            return <ChevronLeftIcon className="h-4 w-4" />;
          } else if (props.orientation === "right") {
            return <ChevronRightIcon className="h-4 w-4" />;
          } else if (props.orientation === "down") {
            return <ChevronDownIcon className="h-4 w-4" />;
          } else {
            return <ChevronUpIcon className="h-4 w-4" />;
          }
        },
        Dropdown: (props) => <CustomDropdown {...props} />,
      }}
      {...props}
    />
  );
}
Calendar.displayName = "Calendar";

const CustomDropdown = ({
  options,
  value,
  onChange,
  name,
  disabled,
}: DropdownProps) => {
  const handleValueChange = (newValue: string) => {
    if (onChange) {
      const syntheticEvent = {
        target: {
          name,
          value: newValue,
        },
      } as React.ChangeEvent<HTMLSelectElement>;
      onChange(syntheticEvent);
    }
  };

  return (
    <Select
      value={value?.toString()}
      onValueChange={handleValueChange}
      disabled={disabled}
    >
      <SelectTrigger className="w-[120px] space-x-2 bg-white text-sm">
        <SelectValue placeholder="Select" />
      </SelectTrigger>
      <SelectContent className="bg-white">
        {options?.map((option) => (
          <SelectItem
            key={option.value}
            value={option.value.toString()}
            disabled={option.disabled}
          >
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export { Calendar };
