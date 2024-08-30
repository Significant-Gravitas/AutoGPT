import React, { FC, useEffect } from "react";
import { Button } from "./ui/button";
import { NodeExecutionResult } from "@/lib/autogpt-server-api/types";
import DataTable from "./DataTable";

interface OutputModalProps {
  isOpen: boolean;
  onClear: () => void;
  onClose: () => void;
  output_data: NodeExecutionResult["output_data"][];
}

const OutputModalComponent: FC<OutputModalProps> = ({
  isOpen,
  onClear,
  onClose,
  output_data,
}) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="nodrag nowheel fixed inset-0 flex items-center justify-center bg-white bg-opacity-60">
      <div className="w-[500px] max-w-[90%] rounded-lg border-[1.5px] bg-white p-5">
        <strong>Output Data History</strong>
        <div className="my-2 max-h-[384px] flex-grow overflow-y-auto rounded-md border-[1.5px] p-2">
          {output_data.map((data, i) => (
            <DataTable key={i} title="Execution" data={data} />
          ))}
        </div>
        <div className="mt-2.5 flex justify-end gap-2.5">
          <Button variant="destructive" onClick={onClear}>
            Clear
          </Button>
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  );
};

export default OutputModalComponent;
