import React from 'react';

interface NodePropertiesProps {
  status: string | undefined;
  outputData: any;
}

export const NodeProperties: React.FC<NodePropertiesProps> = ({ status, outputData }) => {
  return (
    <div className="node-properties">
      <h4>Node Output</h4>
      <p>
        <strong>Status:</strong>{' '}
        {typeof status === 'object' ? JSON.stringify(status) : status || 'N/A'}
      </p>
      <p>
        <strong>Output Data:</strong>{' '}
        {typeof outputData === 'object'
          ? JSON.stringify(outputData)
          : outputData || 'N/A'}
      </p>
    </div>
  );
};