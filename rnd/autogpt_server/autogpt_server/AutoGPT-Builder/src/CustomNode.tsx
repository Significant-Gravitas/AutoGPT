import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';

type Schema = {
  properties: { [key: string]: any };
};

const CustomNode: React.FC<NodeProps> = ({ data }) => {
  const generateHandles = (schema: Schema, type: 'source' | 'target') => {
    if (!schema?.properties) return null;
    const keys = Object.keys(schema.properties);
    return keys.map((key, index) => (
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

  return (
    <div style={{ padding: '20px', border: '2px solid #fff', borderRadius: '20px', background: '#333', color: '#e0e0e0', width: '250px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
          {data?.title.replace(/\d+/g, '')}
        </div>
        <button
          style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#e0e0e0' }}
          onClick={(event: React.MouseEvent) => {
            event.stopPropagation();
            data.openPropertiesSidebar(data);
          }}
        >
          &#9776;
        </button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', gap: '20px' }}>
        <div>
          {data.inputSchema && generateHandles(data.inputSchema, 'target')}
        </div>
        <div>
          {data.outputSchema && generateHandles(data.outputSchema, 'source')}
        </div>
      </div>
    </div>
  );
};

export default CustomNode;
