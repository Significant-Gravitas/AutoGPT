import React from 'react';
import { Button } from '../ui/button';

interface NodeFooterProps {
  toggleProperties: () => void;
  toggleAdvancedSettings: () => void;
  hasOptionalFields: boolean;
}

export const NodeFooter: React.FC<NodeFooterProps> = ({
  toggleProperties,
  toggleAdvancedSettings,
  hasOptionalFields,
}) => {
  return (
    <div className="node-footer">
      <Button onClick={toggleProperties} className="toggle-button">
        Toggle Properties
      </Button>
      {hasOptionalFields && (
        <Button onClick={toggleAdvancedSettings} className="toggle-button">
          Toggle Advanced
        </Button>
      )}
    </div>
  );
};