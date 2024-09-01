import { beautifyString } from "@/lib/utils";
import { Button } from "./ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Clipboard } from "lucide-react";
import { useToast } from "./ui/use-toast";

type DataTableProps = {
  title?: string;
  truncateLongData?: boolean;
  data: { [key: string]: Array<any> };
};

export default function DataTable({
  title,
  truncateLongData,
  data,
}: DataTableProps) {
  const { toast } = useToast();
  const maxChars = 100;

  const copyData = (pin: string, data: string) => {
    navigator.clipboard.writeText(data).then(() => {
      toast({
        title: `"${pin}" data copied to clipboard!`,
        duration: 2000,
      });
    });
  };

  return (
    <>
      {title && <strong className="mt-2 flex justify-center">{title}</strong>}
      <Table className="cursor-default select-text">
        <TableHeader>
          <TableRow>
            <TableHead>Pin</TableHead>
            <TableHead>Data</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.entries(data).map(([key, value]) => (
            <TableRow className="group" key={key}>
              <TableCell className="cursor-text">
                {beautifyString(key)}
              </TableCell>
              <TableCell className="cursor-text">
                <div className="flex min-h-9 items-center">
                  <Button
                    className="absolute right-1 top-auto m-1 hidden p-2 group-hover:block"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      copyData(
                        beautifyString(key),
                        value
                          .map((i) =>
                            typeof i === "object"
                              ? JSON.stringify(i)
                              : String(i),
                          )
                          .join(", "),
                      )
                    }
                    title="Copy Data"
                  >
                    <Clipboard size={18} />
                  </Button>
                  {value
                    .map((i) => {
                      const text =
                        typeof i === "object" ? JSON.stringify(i) : String(i);
                      return truncateLongData && text.length > maxChars
                        ? text.slice(0, maxChars) + "..."
                        : text;
                    })
                    .join(", ")}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  );
}
