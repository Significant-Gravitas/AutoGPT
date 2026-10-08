"use client";

import { Calendar } from "@/components/ui/calendar";

interface Props {
  selected?: Date;
  onSelect: (date?: Date) => void;
}

export function DatePickerCalendar({ selected, onSelect }: Props) {
  return (
    <Calendar
      mode="single"
      selected={selected}
      defaultMonth={selected}
      onSelect={onSelect}
      showOutsideDays
    />
  );
}
