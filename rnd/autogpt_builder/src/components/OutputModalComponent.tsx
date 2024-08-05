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
    <div className="fixed inset-0 bg-white bg-opacity-60 flex justify-center items-center z-50">
      <div className="bg-white p-5 rounded-lg w-[1000px] max-w-[100%]">
        <center>
          <h1 style={{ color: "black" }}>Full Output</h1>
        </center>
        <Textarea
          className="w-full h-[400px] p-2.5 rounded border border-[#dfdfdf] text-black bg-[#dfdfdf]"
          value={tempValue}
          readOnly
        />
        <div className="flex justify-end gap-2.5 mt-2.5">
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>,
    document.body,
  );
};

export default OutputModalComponent;
