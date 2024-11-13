import { Schedule } from "@/lib/autogpt-server-api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  CalendarIcon,
  ClockIcon,
  PlayIcon,
  StopCircleIcon,
} from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface SchedulesTableProps {
  schedules: Schedule[];
  onToggleSchedule: (scheduleId: string, enabled: boolean) => void;
  sortColumn: keyof Schedule;
  sortDirection: "asc" | "desc";
  onSort: (column: keyof Schedule) => void;
}

export const SchedulesTable = ({
  schedules,
  onToggleSchedule,
  sortColumn,
  sortDirection,
  onSort,
}: SchedulesTableProps) => {
  const { toast } = useToast();

  const sortedSchedules = [...schedules].sort((a, b) => {
    const aValue = a[sortColumn];
    const bValue = b[sortColumn];
    if (sortDirection === "asc") {
      return String(aValue).localeCompare(String(bValue));
    }
    return String(bValue).localeCompare(String(aValue));
  });

  const handleToggleSchedule = (scheduleId: string, enabled: boolean) => {
    onToggleSchedule(scheduleId, enabled);
    if (!enabled) {
      toast({
        title: "Schedule Disabled",
        description: "The schedule has been successfully disabled.",
      });
    }
  };

  return (
    <Card className="h-fit p-4">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Schedules</h3>
        <Button size="sm" variant="outline">
          <ClockIcon className="mr-2 h-4 w-4" />
          New Schedule
        </Button>
      </div>
      <ScrollArea className="max-h-[400px]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead
                onClick={() => onSort("agentGraphId")}
                className="cursor-pointer"
              >
                Graph ID
              </TableHead>
              <TableHead
                onClick={() => onSort("schedule")}
                className="cursor-pointer"
              >
                Schedule
              </TableHead>
              <TableHead
                onClick={() => onSort("isEnabled")}
                className="cursor-pointer"
              >
                Status
              </TableHead>
              <TableHead
                onClick={() => onSort("createdAt")}
                className="cursor-pointer"
              >
                Created
              </TableHead>

              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedSchedules.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={6}
                  className="py-8 text-center text-lg text-gray-400"
                >
                  No schedules are available
                </TableCell>
              </TableRow>
            ) : (
              sortedSchedules.map((schedule) => (
                <TableRow key={schedule.id}>
                  <TableCell className="font-medium">
                    {schedule.agentGraphId}
                  </TableCell>
                  <TableCell>{schedule.schedule}</TableCell>
                  <TableCell>
                    <Badge
                      variant={!schedule.isEnabled ? "destructive" : "default"}
                    >
                      {schedule.isEnabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </TableCell>
                  <TableCell>{schedule.createdAt}</TableCell>

                  <TableCell>
                    <div className="flex space-x-2">
                      <Button
                        variant={"destructive"}
                        onClick={() =>
                          handleToggleSchedule(schedule.id, !schedule.isEnabled)
                        }
                      >
                        Disable
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </ScrollArea>
    </Card>
  );
};
