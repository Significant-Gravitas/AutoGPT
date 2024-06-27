import React, { useState, useEffect, FC, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';
import './customnode.css'; // Make sure to create and import this CSS file
import ModalComponent from './ModalComponent'; // Import the modal component

type Schema = {
  properties: { [key: string]: any };
};

const CustomNode: FC<NodeProps> = ({ data, id }) => {
  const [isPropertiesOpen, setIsPropertiesOpen] = useState(data.isPropertiesOpen || false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);

  useEffect(() => {
    if (data.output_data || data.status) {
      setIsPropertiesOpen(true);
    }
  }, [data.output_data, data.status]);

  const toggleProperties = () => {
    setIsPropertiesOpen(!isPropertiesOpen);
  };

  const generateHandles = (schema: Schema, type: 'source' | 'target') => {
    if (!schema?.properties) return null;
    const keys = Object.keys(schema.properties);
    return keys.map((key) => (
      <div key={key} className="handle-container">
        {type === 'target' && (
          <>
            <Handle
              type={type}
              position={Position.Left}
              id={key}
              style={{ background: '#555', borderRadius: '50%' }}
            />
            <span className="handle-label">{key}</span>
          </>
        )}
        {type === 'source' && (
          <>
            <span className="handle-label">{key}</span>
            <Handle
              type={type}
              position={Position.Right}
              id={key}
              style={{ background: '#555', borderRadius: '50%' }}
            />
          </>
        )}
      </div>
    ));
  };

  const handleInputChange = (key: string, value: any) => {
    const newValues = { ...data.hardcodedValues, [key]: value };
    data.setHardcodedValues(newValues);
  };

  const isHandleConnected = (key: string) => {
    return data.connections && data.connections.some((conn: string) => {
      const [source, target] = conn.split(' -> ');
      return target.includes(key) && target.includes(data.title);
    });
  };

  const handleInputClick = (key: string) => {
    setActiveKey(key);
    setIsModalOpen(true);
  };

  const handleModalSave = (value: string) => {
    if (activeKey) {
      handleInputChange(activeKey, value);
    }
    setIsModalOpen(false);
    setActiveKey(null);
  };

  return (
    <div className="custom-node">
      <div className="node-header">
        <div className="node-title">
          {data?.title.replace(/\d+/g, '')}
        </div>
        <button
          onClick={toggleProperties}
          className="toggle-button"
        >
          &#9776;
        </button>
      </div>
      <div className="node-content">
        <div>
          {data.inputSchema && generateHandles(data.inputSchema, 'target')}
          {data.inputSchema && Object.keys(data.inputSchema.properties).map(key => (
            !isHandleConnected(key) && (
              <div key={key} className="input-container">
                <div
                  className="clickable-input"
                  onClick={() => handleInputClick(key)}
                >
                  {data.hardcodedValues[key] || `Enter ${key}`}
                </div>
              </div>
            )
          ))}
        </div>
        <div>
          {data.outputSchema && generateHandles(data.outputSchema, 'source')}
        </div>
      </div>
      {isPropertiesOpen && (
        <div className="node-properties">
          <h4>Node Output</h4>
          <p><strong>Status:</strong> {typeof data.status === 'object' ? JSON.stringify(data.status) : data.status || 'N/A'}</p>
          <p><strong>Output Data:</strong> {typeof data.output_data === 'object' ? JSON.stringify(data.output_data) : data.output_data || 'N/A'}</p>
        </div>
      )}
      <ModalComponent
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        value={activeKey ? data.hardcodedValues[activeKey] : ''}
      />
    </div>
  );
};

export default memo(CustomNode);
