import { Select } from "@/components/atoms/Select/Select";
import { cn } from "@/lib/utils";
import { useId } from "react";

interface Props {
  value?: string;
  onChange: (time: string) => void;
  className?: string;
}

export function TimePicker({ value, onChange, className }: Props) {
  const id = useId();
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
    <div className={cn("flex items-center space-x-3", className)}>
      <div className="flex flex-col items-center">
        <Select
          id={`${id}-hour`}
          label="Hour"
          hideLabel
          size="md"
          value={hour}
          onValueChange={(val: string) => changeTime(val, minute, meridiem)}
          options={Array.from({ length: 12 }, (_, i) => pad(i + 1)).map(
            (h) => ({
              value: h,
              label: h,
            }),
          )}
        />
      </div>

      <div className="mb-6 flex flex-col items-center">
        <span className="m-auto text-xl font-bold">:</span>
      </div>

      <div className="flex flex-col items-center">
        <Select
          id={`${id}-minute`}
          label="Minute"
          hideLabel
          size="md"
          value={minute}
          onValueChange={(val: string) => changeTime(hour, val, meridiem)}
          options={Array.from({ length: 60 }, (_, i) => pad(i)).map((m) => ({
            value: m.toString(),
            label: m,
          }))}
        />
      </div>

      <div className="flex flex-col items-center">
        <Select
          id={`${id}-meridiem`}
          label="AM/PM"
          hideLabel
          size="md"
          value={meridiem}
          onValueChange={(val: string) => changeTime(hour, minute, val)}
          options={[
            { value: "AM", label: "AM" },
            { value: "PM", label: "PM" },
          ]}
        />
      </div>
    </div>
  );
}
