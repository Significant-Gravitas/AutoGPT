import { beautifyString } from "@/lib/utils";
import { Clipboard, Maximize2 } from "lucide-react";
import React, { useState } from "react";
import { Button } from "../../../../../components/__legacy__/ui/button";
import { ContentRenderer } from "../../../../../components/__legacy__/ui/render";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../../../../components/__legacy__/ui/table";
import { useToast } from "../../../../../components/molecules/Toast/use-toast";
import ExpandableOutputDialog from "./ExpandableOutputDialog";

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
  const [expandedDialog, setExpandedDialog] = useState<{
    isOpen: boolean;
    execId: string;
    pinName: string;
    data: any[];
  } | null>(null);

  const copyData = (pin: string, data: string) => {
    navigator.clipboard.writeText(data).then(() => {
      toast({
        title: `"${pin}" data copied to clipboard!`,
        duration: 2000,
      });
    });
  };

  const openExpandedView = (pinName: string, pinData: any[]) => {
    setExpandedDialog({
      isOpen: true,
      execId: title || "Unknown Execution",
      pinName,
      data: pinData,
    });
  };

  const closeExpandedView = () => {
    setExpandedDialog(null);
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
                <div className="flex min-h-9 items-center whitespace-pre-wrap">
                  <div className="absolute right-1 top-auto m-1 hidden gap-1 group-hover:flex">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => openExpandedView(key, value)}
                      title="Expand Full View"
                    >
                      <Maximize2 size={18} />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() =>
                        copyData(
                          beautifyString(key),
                          value
                            .map((i) =>
                              typeof i === "object"
                                ? JSON.stringify(i, null, 2)
                                : String(i),
                            )
                            .join(", "),
                        )
                      }
                      title="Copy Data"
                    >
                      <Clipboard size={18} />
                    </Button>
                  </div>
                  {value.map((item, index) => (
                    <React.Fragment key={index}>
                      <ContentRenderer
                        value={item}
                        truncateLongData={truncateLongData}
                      />
                      {index < value.length - 1 && ", "}
                    </React.Fragment>
                  ))}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {expandedDialog && (
        <ExpandableOutputDialog
          isOpen={expandedDialog.isOpen}
          onClose={closeExpandedView}
          execId={expandedDialog.execId}
          pinName={expandedDialog.pinName}
          data={expandedDialog.data}
        />
      )}
    </>
  );
}
