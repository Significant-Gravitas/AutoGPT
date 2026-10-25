"use client";

import * as React from "react";
import {
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
} from "@radix-ui/react-icons";
import {
  CaptionLabel,
  Day,
  DayButton,
  DayPicker,
  DropdownNav,
  DropdownProps,
  Footer,
  Month,
  MonthCaption,
  MonthGrid,
  Months,
  Nav,
  NextMonthButton,
  PreviousMonthButton,
  Root,
  Week,
  Weekday,
  Weekdays,
  Weeks,
} from "react-day-picker";

import { cn } from "@/lib/utils";
import { legacyButtonVariants } from "@/components/atoms/Button/helpers";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/atoms/Select/components/SelectPrimitives";

type CalendarProps = React.ComponentProps<typeof DayPicker>;

// DayPicker's own parts, unchanged except for a data-slot on the element each
// renders. The slot goes before the props so a caller's data-slot still wins.
function withSlot<P extends object>(
  Part: React.ComponentType<P>,
  slot: string,
) {
  function SlottedPart(props: P) {
    return <Part data-slot={slot} {...props} />;
  }
  return SlottedPart;
}

const slottedParts = {
  Root: withSlot(Root, "calendar"),
  Months: withSlot(Months, "calendar-months"),
  Month: withSlot(Month, "calendar-month"),
  MonthCaption: withSlot(MonthCaption, "calendar-month-caption"),
  DropdownNav: withSlot(DropdownNav, "calendar-dropdowns"),
  CaptionLabel: withSlot(CaptionLabel, "calendar-caption-label"),
  Nav: withSlot(Nav, "calendar-nav"),
  PreviousMonthButton: withSlot(PreviousMonthButton, "calendar-previous"),
  NextMonthButton: withSlot(NextMonthButton, "calendar-next"),
  MonthGrid: withSlot(MonthGrid, "calendar-month-grid"),
  Weekdays: withSlot(Weekdays, "calendar-weekdays"),
  Weekday: withSlot(Weekday, "calendar-weekday"),
  Weeks: withSlot(Weeks, "calendar-weeks"),
  Week: withSlot(Week, "calendar-week"),
  Day: withSlot(Day, "calendar-day"),
  DayButton: withSlot(DayButton, "calendar-day-button"),
  Footer: withSlot(Footer, "calendar-footer"),
};

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
        month_caption:
          "flex  items-center justify-center text-body leading-5 font-medium",
        dropdowns: "flex gap-2 ",
        caption_label: "hidden",
        nav: "hidden",
        button_previous:
          "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100",
        button_next: "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100",
        month_grid: "w-full border-collapse space-y-1",
        weekdays: "flex",
        // TODO(tokens): no exact role: text-neutral-500 #737373. SECRT-3152
        // rare size: text-[0.8rem].
        weekday:
          "text-neutral-500 rounded-md w-8 font-normal text-[0.8rem] dark:text-neutral-400",
        week: "flex w-full mt-2",
        // TODO(tokens): no exact role: bg-neutral-100 #F5F5F5.
        day: cn(
          "relative p-0 text-center text-body leading-5 focus-within:relative focus-within:z-20 [&:has([aria-selected])]:bg-neutral-100 [&:has([aria-selected].day-outside)]:bg-neutral-100/50 [&:has([aria-selected].day-range-end)]:rounded-r-md dark:[&:has([aria-selected])]:bg-neutral-800 dark:[&:has([aria-selected].day-outside)]:bg-neutral-800/50",
          props.mode === "range"
            ? "[&:has(>.day-range-end)]:rounded-r-md [&:has(>.day-range-start)]:rounded-l-md first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md"
            : "[&:has([aria-selected])]:rounded-md",
        ),
        day_button: cn(
          legacyButtonVariants({ variant: "legacy-ghost" }),
          "h-8 w-8 p-0 font-normal aria-selected:opacity-100",
        ),
        range_start: "range-start",
        range_end: "range-end",
        // TODO(tokens): no exact role: bg-neutral-900 #171717,
        // text-neutral-100 #F5F5F5, bg-neutral-700 #404040. text-neutral-50
        // #FAFAFA is `primary-foreground` in :root only (#18181B in .dark),
        // so it stays raw. SECRT-3154
        selected:
          "bg-neutral-900 text-neutral-100 hover:bg-neutral-900 hover:text-neutral-50 focus:bg-neutral-700 focus:text-neutral-50 dark:bg-neutral-50 dark:text-neutral-900 dark:hover:bg-neutral-50 dark:hover:text-neutral-900 dark:focus:bg-neutral-50 dark:focus:text-neutral-900",
        // TODO(tokens): no exact role: bg-neutral-100 #F5F5F5,
        // text-neutral-900 #171717.
        today:
          "bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-50",
        // TODO(tokens): no exact role: text-neutral-500 #737373,
        // bg-neutral-100 #F5F5F5.
        outside:
          "day-outside text-neutral-500 opacity-50  aria-selected:bg-neutral-100/50 aria-selected:text-neutral-500 aria-selected:opacity-30 dark:text-neutral-400 dark:aria-selected:bg-neutral-800/50 dark:aria-selected:text-neutral-400",
        // TODO(tokens): no exact role: text-neutral-500 #737373.
        disabled: "text-neutral-500 opacity-50 dark:text-neutral-400",
        // TODO(tokens): no exact role: bg-neutral-100 #F5F5F5,
        // text-neutral-900 #171717.
        range_middle:
          "aria-selected:bg-neutral-100 aria-selected:text-neutral-900 dark:aria-selected:bg-neutral-800 dark:aria-selected:text-neutral-50",
        hidden: "invisible",
        ...classNames,
      }}
      components={{
        ...slottedParts,
        Chevron: (props) => {
          if (props.orientation === "left") {
            return (
              <ChevronLeftIcon
                data-slot="calendar-chevron"
                className="h-4 w-4"
              />
            );
          } else if (props.orientation === "right") {
            return (
              <ChevronRightIcon
                data-slot="calendar-chevron"
                className="h-4 w-4"
              />
            );
          } else if (props.orientation === "down") {
            return (
              <ChevronDownIcon
                data-slot="calendar-chevron"
                className="h-4 w-4"
              />
            );
          } else {
            return (
              <ChevronUpIcon data-slot="calendar-chevron" className="h-4 w-4" />
            );
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
      <SelectTrigger
        data-slot="calendar-dropdown-trigger"
        className="w-[120px] space-x-2 bg-surface"
      >
        <SelectValue data-slot="calendar-dropdown-value" placeholder="Select" />
      </SelectTrigger>
      <SelectContent data-slot="calendar-dropdown-popup" className="bg-surface">
        {options?.map((option) => (
          <SelectItem
            key={option.value}
            data-slot="calendar-dropdown-item"
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
