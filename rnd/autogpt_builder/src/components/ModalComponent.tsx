import React, { FC } from 'react';
import './modal.css';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (value: string) => void;
  value: string;
}

const ModalComponent: FC<ModalProps> = ({ isOpen, onClose, onSave, value }) => {
  const [tempValue, setTempValue] = React.useState(value);

  const handleSave = () => {
    onSave(tempValue);
    onClose();
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay">
      <div className="modal">
        <textarea
          className="modal-textarea"
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
        />
        <div className="modal-actions">
          <button onClick={onClose}>Cancel</button>
          <button onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
};

export default ModalComponent;
