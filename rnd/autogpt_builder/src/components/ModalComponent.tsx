import React, { FC, useEffect } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (value: string) => void;
  value: string;
}

const ModalComponent: FC<ModalProps> = ({ isOpen, onClose, onSave, value }) => {
  const [tempValue, setTempValue] = React.useState(value);

  useEffect(() => {
    if (isOpen) {
      setTempValue(value);
    }
  }, [isOpen, value]);

  const handleSave = () => {
    onSave(tempValue);
    onClose();
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-white bg-opacity-60 flex justify-center items-center">
      <div className="bg-white p-5 rounded-lg w-[500px] max-w-[90%]">
        <center><h1>Enter input text</h1></center>
        <Textarea
          className="w-full h-[200px] p-2.5 rounded border border-[#dfdfdf] text-black bg-[#dfdfdf]"
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
        />
        <div className="flex justify-end gap-2.5 mt-2.5">
          <Button onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave}>Save</Button>
        </div>
      </div>
    </div>
  );
};

export default ModalComponent;
