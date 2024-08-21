import React, { FC, useEffect } from "react";
import { createPortal } from "react-dom";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

interface OutputModalProps {
  isOpen: boolean;
  onClose: () => void;
  value: string;
}

const OutputModalComponent: FC<OutputModalProps> = ({
  isOpen,
  onClose,
  value,
}) => {
  const [tempValue, setTempValue] = React.useState(value);

  useEffect(() => {
    if (isOpen) {
      setTempValue(value);
    }
  }, [isOpen, value]);

  if (!isOpen) {
    return null;
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-white bg-opacity-60">
      <div className="w-[1000px] max-w-[100%] rounded-lg bg-white p-5">
        <center>
          <h1 style={{ color: "black" }}>Full Output</h1>
        </center>
        <Textarea
          className="h-[400px] w-full rounded border border-[#dfdfdf] bg-[#dfdfdf] p-2.5 text-black"
          value={tempValue}
          readOnly
        />
        <div className="mt-2.5 flex justify-end gap-2.5">
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>,
    document.body,
  );
};

export default OutputModalComponent;
