import React from 'react';
import { Handle, Position } from 'reactflow';
import { CustomNodeData } from './types';
import { InputField } from './InputField';
import SchemaTooltip from '../SchemaTooltip';
import { isHandleConnected } from './utils';
import { beautifyString } from '@/lib/utils';

type NodeContentProps = {
  data: CustomNodeData;
  isAdvancedOpen: boolean;
  handleInputClick: (key: string) => void;
};

export const NodeContent: React.FC<NodeContentProps> = ({ data, isAdvancedOpen, handleInputClick }) => {
  return (
    <div className="node-content">
      <div className="input-section">
        {data.inputSchema &&
          Object.entries(data.inputSchema.properties).map(([key, schema]) => {
            const isRequired = data.inputSchema.required?.includes(key);
            return (isRequired || isAdvancedOpen) && (
              <div key={key}>
                <div className="handle-container">
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={key}
                    style={{ background: '#555', borderRadius: '50%', width: '10px', height: '10px' }}
                  />
                  <span className="handle-label">{schema.title || beautifyString(key)}</span>
                  <SchemaTooltip schema={schema} />
                </div>
                <InputField
                  schema={schema}
                  fullKey={key}
                  displayKey={schema.title || beautifyString(key)}
                  value={data.hardcodedValues[key]}
                  error={null}
                  handleInputChange={(key, value) => data.setHardcodedValues({ ...data.hardcodedValues, [key]: value })}
                  handleInputClick={handleInputClick}
                />
              </div>
            );
          })}
      </div>
      <div className="output-section">
        {data.outputSchema && generateHandles(data.outputSchema, 'source')}
      </div>
    </div>
  );
};