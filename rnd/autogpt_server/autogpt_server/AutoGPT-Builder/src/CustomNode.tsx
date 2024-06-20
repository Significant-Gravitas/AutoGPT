import React, { useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';

type Schema = {
  properties: { [key: string]: any };
};

const CustomNode: React.FC<NodeProps> = ({ data }) => {
  const [isPropertiesOpen, setIsPropertiesOpen] = useState(false);

  const toggleProperties = () => {
    setIsPropertiesOpen(!isPropertiesOpen);
  };

  const generateHandles = (schema: Schema, type: 'source' | 'target') => {
    if (!schema?.properties) return null;
    const keys = Object.keys(schema.properties);
    return keys.map((key) => (
      <div key={key} style={{ display: 'flex', alignItems: 'center', position: 'relative', marginBottom: '5px' }}>
        {type === 'target' && (
          <>
            <Handle
              type={type}
              position={Position.Left}
              id={`${type}-${key}`}
              style={{ background: '#555', borderRadius: '50%' }}
            />
            <span style={{ color: '#e0e0e0', marginLeft: '10px' }}>{key}</span>
          </>
        )}
        {type === 'source' && (
          <>
            <span style={{ color: '#e0e0e0', marginRight: '10px' }}>{key}</span>
            <Handle
              type={type}
              position={Position.Right}
              id={`${type}-${key}`}
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
    return data.connections.some((conn: string) => {
      const [source, target] = conn.split(' -> ');
      return target.includes(key) && target.includes(data.title);
    });
  };

  return (
    <div style={{ padding: '20px', border: '2px solid #fff', borderRadius: '20px', background: '#333', color: '#e0e0e0', width: '250px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
          {data?.title.replace(/\d+/g, '')}
        </div>
        <button
          onClick={toggleProperties}
          style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#e0e0e0' }}
        >
          &#9776;
        </button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', gap: '20px' }}>
        <div>
          {data.inputSchema && generateHandles(data.inputSchema, 'target')}
          {data.inputSchema && Object.keys(data.inputSchema.properties).map(key => (
            (!isHandleConnected(key) || data.connections.length === 0) && (
              <div key={key} style={{ marginBottom: '5px' }}>
                <input
                  type="text"
                  placeholder={`Enter ${key}`}
                  value={data.hardcodedValues[key] || ''}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                  style={{ width: '100%', padding: '5px', borderRadius: '4px', border: '1px solid #555', background: '#444', color: '#e0e0e0' }}
                />
              </div>
            )
          ))}
        </div>
        <div>
          {data.outputSchema && generateHandles(data.outputSchema, 'source')}
        </div>
      </div>
      {isPropertiesOpen && (
        <div style={{ marginTop: '10px', background: '#444', padding: '10px', borderRadius: '10px' }}>
          <h4>Node Properties</h4>
          <p><strong>Name:</strong> {data.title}</p>
          <p><strong>Description:</strong> {data.description}</p>
          {/* Add more properties here if needed */}
        </div>
      )}
    </div>
  );
};

export default CustomNode;
