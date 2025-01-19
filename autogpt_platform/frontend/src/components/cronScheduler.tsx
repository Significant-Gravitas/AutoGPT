import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Separator } from "./ui/separator";
import { CronExpressionManager } from "@/lib/monitor/cronExpressionManager";

interface CronSchedulerProps {
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  open: boolean;
  afterCronCreation: (cronExpression: string) => void;
}

export function CronScheduler({
  setOpen,
  open,
  afterCronCreation,
}: CronSchedulerProps) {
  const [frequency, setFrequency] = useState<
    "minute" | "hour" | "daily" | "weekly" | "monthly" | "yearly" | "custom"
  >("minute");
  const [selectedDays, setSelectedDays] = useState<number[]>([]);
  const [selectedTime, setSelectedTime] = useState<string>("00:00");
  const [showCustomDays, setShowCustomDays] = useState<boolean>(false);
  const [selectedMinute, setSelectedMinute] = useState<string>("0");
  const [customInterval, setCustomInterval] = useState<{
    value: number;
    unit: "minutes" | "hours" | "days";
  }>({ value: 1, unit: "minutes" });

  // const [endType, setEndType] = useState<"never" | "on" | "after">("never");
  // const [endDate, setEndDate] = useState<Date | undefined>();
  // const [occurrences, setOccurrences] = useState<number>(1);

  const weekDays = [
    { label: "Su", value: 0 },
    { label: "Mo", value: 1 },
    { label: "Tu", value: 2 },
    { label: "We", value: 3 },
    { label: "Th", value: 4 },
    { label: "Fr", value: 5 },
    { label: "Sa", value: 6 },
  ];

  const months = [
    { label: "Jan", value: "January" },
    { label: "Feb", value: "February" },
    { label: "Mar", value: "March" },
    { label: "Apr", value: "April" },
    { label: "May", value: "May" },
    { label: "Jun", value: "June" },
    { label: "Jul", value: "July" },
    { label: "Aug", value: "August" },
    { label: "Sep", value: "September" },
    { label: "Oct", value: "October" },
    { label: "Nov", value: "November" },
    { label: "Dec", value: "December" },
  ];

  const cron_manager = new CronExpressionManager();

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogTitle>Schedule Task</DialogTitle>
        <div className="max-w-md space-y-6 p-2">
          <div className="space-y-4">
            <Label className="text-base font-medium">Repeat</Label>

            <Select
              onValueChange={(value: any) => setFrequency(value)}
              defaultValue="minute"
            >
              <SelectTrigger>
                <SelectValue placeholder="Select frequency" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="minute">Every Minute</SelectItem>
                <SelectItem value="hour">Every Hour</SelectItem>
                <SelectItem value="daily">Daily</SelectItem>
                <SelectItem value="weekly">Weekly</SelectItem>
                <SelectItem value="monthly">Monthly</SelectItem>
                <SelectItem value="yearly">Yearly</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>

            {frequency === "hour" && (
              <div className="flex items-center gap-2">
                <Label>At minute</Label>
                <Select
                  value={selectedMinute}
                  onValueChange={setSelectedMinute}
                >
                  <SelectTrigger className="w-24">
                    <SelectValue placeholder="Select minute" />
                  </SelectTrigger>
                  <SelectContent>
                    {[0, 15, 30, 45].map((min) => (
                      <SelectItem key={min} value={min.toString()}>
                        {min}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {frequency === "custom" && (
              <div className="flex items-center gap-2">
                <Label>Every</Label>
                <Input
                  type="number"
                  min="1"
                  className="w-20"
                  value={customInterval.value}
                  onChange={(e) =>
                    setCustomInterval({
                      ...customInterval,
                      value: parseInt(e.target.value),
                    })
                  }
                />
                <Select
                  value={customInterval.unit}
                  onValueChange={(value: any) =>
                    setCustomInterval({ ...customInterval, unit: value })
                  }
                >
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="minutes">Minutes</SelectItem>
                    <SelectItem value="hours">Hours</SelectItem>
                    <SelectItem value="days">Days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          {frequency === "weekly" && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label>On</Label>
                <Button
                  variant="outline"
                  className="h-8 px-2 py-1 text-xs"
                  onClick={() => {
                    if (selectedDays.length === weekDays.length) {
                      setSelectedDays([]);
                    } else {
                      setSelectedDays(weekDays.map((day) => day.value));
                    }
                  }}
                >
                  {selectedDays.length === weekDays.length
                    ? "Deselect All"
                    : "Select All"}
                </Button>
                <Button
                  variant="outline"
                  className="h-8 px-2 py-1 text-xs"
                  onClick={() => setSelectedDays([1, 2, 3, 4, 5])}
                >
                  Weekdays
                </Button>
                <Button
                  variant="outline"
                  className="h-8 px-2 py-1 text-xs"
                  onClick={() => setSelectedDays([0, 6])}
                >
                  Weekends
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {weekDays.map((day) => (
                  <Button
                    key={day.value}
                    variant={
                      selectedDays.includes(day.value) ? "default" : "outline"
                    }
                    className="h-10 w-10 p-0"
                    onClick={() => {
                      setSelectedDays((prev) =>
                        prev.includes(day.value)
                          ? prev.filter((d) => d !== day.value)
                          : [...prev, day.value],
                      );
                    }}
                  >
                    {day.label}
                  </Button>
                ))}
              </div>
            </div>
          )}
          {frequency === "monthly" && (
            <div className="space-y-4">
              <Label>Days of Month</Label>
              <div className="flex gap-2">
                <Button
                  variant={!showCustomDays ? "default" : "outline"}
                  onClick={() => {
                    setShowCustomDays(false);
                    setSelectedDays(
                      Array.from({ length: 31 }, (_, i) => i + 1),
                    );
                  }}
                >
                  All Days
                </Button>
                <Button
                  variant={showCustomDays ? "default" : "outline"}
                  onClick={() => {
                    setShowCustomDays(true);
                    setSelectedDays([]);
                  }}
                >
                  Customize
                </Button>
                <Button variant="outline" onClick={() => setSelectedDays([15])}>
                  15th
                </Button>
                <Button variant="outline" onClick={() => setSelectedDays([31])}>
                  Last Day
                </Button>
              </div>
              {showCustomDays && (
                <div className="flex flex-wrap gap-2">
                  {Array.from({ length: 31 }, (_, i) => (
                    <Button
                      key={i + 1}
                      variant={
                        selectedDays.includes(i + 1) ? "default" : "outline"
                      }
                      className="h-10 w-10 p-0"
                      onClick={() => {
                        setSelectedDays((prev) =>
                          prev.includes(i + 1)
                            ? prev.filter((d) => d !== i + 1)
                            : [...prev, i + 1],
                        );
                      }}
                    >
                      {i + 1}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}
          {frequency === "yearly" && (
            <div className="space-y-4">
              <Label>Months</Label>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="h-8 px-2 py-1 text-xs"
                  onClick={() => {
                    if (selectedDays.length === months.length) {
                      setSelectedDays([]);
                    } else {
                      setSelectedDays(Array.from({ length: 12 }, (_, i) => i));
                    }
                  }}
                >
                  {selectedDays.length === months.length
                    ? "Deselect All"
                    : "Select All"}
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {months.map((month, index) => (
                  <Button
                    key={index}
                    variant={
                      selectedDays.includes(index) ? "default" : "outline"
                    }
                    className="px-2 py-1"
                    onClick={() => {
                      setSelectedDays((prev) =>
                        prev.includes(index)
                          ? prev.filter((m) => m !== index)
                          : [...prev, index],
                      );
                    }}
                  >
                    {month.label}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {frequency !== "minute" && frequency !== "hour" && (
            <div className="flex items-center gap-4 space-y-2">
              <Label className="pt-2">At</Label>
              <Input
                type="time"
                value={selectedTime}
                onChange={(e) => setSelectedTime(e.target.value)}
              />
            </div>
          )}

          <Separator />
          {/*

            On the backend, we are using standard cron expressions,
            which makes it challenging to add an end date or stop execution
            after a certain time using only cron expressions.
            (since standard cron expressions have limitations, like the lack of a year field or more...).

            We could also use ranges in cron expression for end dates but It doesm't cover all cases (sometimes break)

            To automatically end the scheduler, we need to store the end date and time occurrence in the database
            and modify scheduler.add_job. Currently, we can only stop the scheduler manually from the monitor tab.

            */}

          {/* <div className="space-y-6">
            <Label className="text-lg font-medium">Ends</Label>
            <RadioGroup
              value={endType}
              onValueChange={(value: "never" | "on" | "after") =>
                setEndType(value)
              }
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="never" id="never" />
                <Label htmlFor="never">Never</Label>
              </div>

              <div className="flex items-center space-x-2">
                <RadioGroupItem value="on" id="on" />
                <Label htmlFor="on" className="w-[50px]">
                  On
                </Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className="w-full"
                      disabled={endType !== "on"}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {endDate ? format(endDate, "PPP") : "Pick a date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar
                      mode="single"
                      selected={endDate}
                      onSelect={setEndDate}
                      disabled={(date) => date < new Date()}
                      fromDate={new Date()}
                    />
                  </PopoverContent>
                </Popover>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="after" id="after" />
                <Label htmlFor="after" className="ml-2 w-[50px]">
                  After
                </Label>
                <Input
                  type="number"
                  className="ml-2 w-[100px]"
                  value={occurrences}
                  onChange={(e) => setOccurrences(Number(e.target.value))}
                />
                <span>times</span>
              </div>
            </RadioGroup>
          </div> */}

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                const cronExpr = cron_manager.generateCronExpression(
                  frequency,
                  selectedTime,
                  selectedDays,
                  selectedMinute,
                  customInterval,
                );
                setFrequency("minute");
                setSelectedDays([]);
                setSelectedTime("00:00");
                setShowCustomDays(false);
                setSelectedMinute("0");
                setCustomInterval({ value: 1, unit: "minutes" });
                setOpen(false);
                afterCronCreation(cronExpr);
              }}
            >
              Done
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
