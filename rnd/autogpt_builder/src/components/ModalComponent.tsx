import React, { FC, useEffect } from 'react';
import './modal.css';
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
    <div className="modal-overlay">
      <div className="modal dark-theme">
        <center><h1>Enter input text</h1></center>
        <Textarea
          className="modal-textarea"
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
        />
        <div className="modal-actions">
          <Button onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave}>Save</Button>
        </div>
      </div>
    </div>
  );
};

export default ModalComponent;
