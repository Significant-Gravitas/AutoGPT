import React, { useState, useEffect, FC, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';
import './customnode.css';
import ModalComponent from './ModalComponent';
import { Button } from './ui/button';
import { Input } from './ui/input';

type Schema = {
  type: string;
  properties: { [key: string]: any };
  required?: string[];
  enum?: string[];
  items?: Schema;
  additionalProperties?: { type: string };
  allOf?: any[];
  anyOf?: any[];
  oneOf?: any[];
};

type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: Schema;
  outputSchema: Schema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>;
  isPropertiesOpen: boolean;
  status?: string;
  output_data?: any;
};

const CustomNode: FC<NodeProps<CustomNodeData>> = ({ data, id }) => {
  const [isPropertiesOpen, setIsPropertiesOpen] = useState(data.isPropertiesOpen || false);
  const [keyValuePairs, setKeyValuePairs] = useState<{ key: string, value: string }[]>([]);
  const [newKey, setNewKey] = useState<string>('');
  const [newValue, setNewValue] = useState<string>('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [modalValue, setModalValue] = useState<string>('');
  const [errors, setErrors] = useState<{ [key: string]: string | null }>({});

  useEffect(() => {
    if (data.output_data || data.status) {
      setIsPropertiesOpen(true);
    }
  }, [data.output_data, data.status]);

  useEffect(() => {
    console.log(`Node ${id} data:`, data);
  }, [id, data]);

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
    const keys = key.split('.');
    const newValues = JSON.parse(JSON.stringify(data.hardcodedValues));
    let current = newValues;

    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) current[keys[i]] = {};
      current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = value;

    console.log(`Updating hardcoded values for node ${id}:`, newValues);
    data.setHardcodedValues(newValues);
    setErrors((prevErrors) => ({ ...prevErrors, [key]: null }));
  };

  const getValue = (key: string) => {
    const keys = key.split('.');
    return keys.reduce((acc, k) => (acc && acc[k] !== undefined) ? acc[k] : '', data.hardcodedValues);
  };

  const isHandleConnected = (key: string) => {
    return data.connections && data.connections.some((conn: any) => {
      if (typeof conn === 'string') {
        const [source, target] = conn.split(' -> ');
        return target.includes(key) && target.includes(data.title);
      }
      return conn.target === id && conn.targetHandle === key;
    });
  };

  const handleAddProperty = () => {
    if (newKey && newValue) {
      const newPairs = [...keyValuePairs, { key: newKey, value: newValue }];
      setKeyValuePairs(newPairs);
      setNewKey('');
      setNewValue('');
      const expectedFormat = newPairs.reduce((acc, pair) => ({ ...acc, [pair.key]: pair.value }), {});
      handleInputChange('expected_format', expectedFormat);
    }
  };

  const handleInputClick = (key: string) => {
    setActiveKey(key);
    const value = getValue(key);
    setModalValue(typeof value === 'object' ? JSON.stringify(value, null, 2) : value);
    setIsModalOpen(true);
  };

  const handleModalSave = (value: string) => {
    if (activeKey) {
      try {
        const parsedValue = JSON.parse(value);
        handleInputChange(activeKey, parsedValue);
      } catch (error) {
        handleInputChange(activeKey, value);
      }
    }
    setIsModalOpen(false);
    setActiveKey(null);
  };

  const renderInputField = (key: string, schema: any, parentKey: string = ''): JSX.Element => {
    const fullKey = parentKey ? `${parentKey}.${key}` : key;
    const error = errors[fullKey];
    const value = getValue(fullKey);

    if (isHandleConnected(fullKey)) {
      return <div className="connected-input">Connected</div>;
    }

    const renderClickableInput = (displayValue: string) => (
      <div className="clickable-input" onClick={() => handleInputClick(fullKey)}>
        {displayValue}
      </div>
    );

    if (schema.type === 'object' && schema.properties) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{key}:</strong>
          {Object.entries(schema.properties).map(([propKey, propSchema]: [string, any]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              {renderInputField(propKey, propSchema, fullKey)}
            </div>
          ))}
        </div>
      );
    }

    if (schema.type === 'object' && schema.additionalProperties) {
      const objectValue = value || {};
      return (
        <div key={fullKey} className="object-input">
          <strong>{key}:</strong>
          {Object.entries(objectValue).map(([propKey, propValue]: [string, any]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              <div className="clickable-input" onClick={() => handleInputClick(`${fullKey}.${propKey}`)}>
                {propKey}: {typeof propValue === 'object' ? JSON.stringify(propValue, null, 2) : propValue}
              </div>
              <Button onClick={() => handleInputChange(`${fullKey}.${propKey}`, undefined)} className="array-item-remove">
                &times;
              </Button>
            </div>
          ))}
          {key === 'expected_format' && (
            <div className="nested-input">
              {keyValuePairs.map((pair, index) => (
                <div key={index} className="key-value-input">
                  <Input
                    type="text"
                    placeholder="Key"
                    value={pair.key}
                    onChange={(e) => {
                      const newPairs = [...keyValuePairs];
                      newPairs[index].key = e.target.value;
                      setKeyValuePairs(newPairs);
                      const expectedFormat = newPairs.reduce((acc, pair) => ({ ...acc, [pair.key]: pair.value }), {});
                      handleInputChange('expected_format', expectedFormat);
                    }}
                  />
                  <Input
                    type="text"
                    placeholder="Value"
                    value={pair.value}
                    onChange={(e) => {
                      const newPairs = [...keyValuePairs];
                      newPairs[index].value = e.target.value;
                      setKeyValuePairs(newPairs);
                      const expectedFormat = newPairs.reduce((acc, pair) => ({ ...acc, [pair.key]: pair.value }), {});
                      handleInputChange('expected_format', expectedFormat);
                    }}
                  />
                </div>
              ))}
              <div className="key-value-input">
                <Input
                  type="text"
                  placeholder="Key"
                  value={newKey}
                  onChange={(e) => setNewKey(e.target.value)}
                />
                <Input
                  type="text"
                  placeholder="Value"
                  value={newValue}
                  onChange={(e) => setNewValue(e.target.value)}
                />
              </div>
              <Button onClick={handleAddProperty}>Add Property</Button>
            </div>
          )}
          {error && <span className="error-message">{error}</span>}
        </div>
      );
    }

    if (schema.anyOf) {
      const types = schema.anyOf.map((s: any) => s.type);
      if (types.includes('string') && types.includes('null')) {
        return (
          <div key={fullKey} className="input-container">
            {renderClickableInput(value || `Enter ${key} (optional)`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      }
    }

    if (schema.allOf) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{key}:</strong>
          {schema.allOf[0].properties && Object.entries(schema.allOf[0].properties).map(([propKey, propSchema]: [string, any]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              {renderInputField(propKey, propSchema, fullKey)}
            </div>
          ))}
        </div>
      );
    }

    if (schema.oneOf) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{key}:</strong>
          {schema.oneOf[0].properties && Object.entries(schema.oneOf[0].properties).map(([propKey, propSchema]: [string, any]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              {renderInputField(propKey, propSchema, fullKey)}
            </div>
          ))}
        </div>
      );
    }

    switch (schema.type) {
      case 'string':
        return schema.enum ? (
          <div key={fullKey} className="input-container">
            <select
              value={value || ''}
              onChange={(e) => handleInputChange(fullKey, e.target.value)}
              className="select-input"
            >
              <option value="">Select {key}</option>
              {schema.enum.map((option: string) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {error && <span className="error-message">{error}</span>}
          </div>
        ) : (
          <div key={fullKey} className="input-container">
            {renderClickableInput(value || `Enter ${key}`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'boolean':
        return (
          <div key={fullKey} className="input-container">
            <select
              value={value === undefined ? '' : value.toString()}
              onChange={(e) => handleInputChange(fullKey, e.target.value === 'true')}
              className="select-input"
            >
              <option value="">Select {key}</option>
              <option value="true">True</option>
              <option value="false">False</option>
            </select>
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'number':
      case 'integer':
        return (
          <div key={fullKey} className="input-container">
            <input
              type="number"
              value={value || ''}
              onChange={(e) => handleInputChange(fullKey, parseFloat(e.target.value))}
              className="number-input"
            />
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'array':
        if (schema.items && schema.items.type === 'string') {
          const arrayValues = value || [];
          return (
            <div key={fullKey} className="input-container">
              {arrayValues.map((item: string, index: number) => (
                <div key={`${fullKey}.${index}`} className="array-item-container">
                  <input
                    type="text"
                    value={item}
                    onChange={(e) => handleInputChange(`${fullKey}.${index}`, e.target.value)}
                    className="array-item-input"
                  />
                  <Button onClick={() => handleInputChange(`${fullKey}.${index}`, '')} className="array-item-remove">
                    &times;
                  </Button>
                </div>
              ))}
              <Button onClick={() => handleInputChange(fullKey, [...arrayValues, ''])} className="array-item-add">
                Add Item
              </Button>
              {error && <span className="error-message">{error}</span>}
            </div>
          );
        }
        return null;
      default:
        return (
          <div key={fullKey} className="input-container">
            {renderClickableInput(value ? `${key} (Complex)` : `Enter ${key} (Complex)`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
    }
  };

  const validateInputs = () => {
    const newErrors: { [key: string]: string | null } = {};
    const validateRecursive = (schema: any, parentKey: string = '') => {
      Object.entries(schema.properties).forEach(([key, propSchema]: [string, any]) => {
        const fullKey = parentKey ? `${parentKey}.${key}` : key;
        const value = getValue(fullKey);

        if (propSchema.type === 'object' && propSchema.properties) {
          validateRecursive(propSchema, fullKey);
        } else {
          if (propSchema.required && !value) {
            newErrors[fullKey] = `${fullKey} is required`;
          }
        }
      });
    };

    validateRecursive(data.inputSchema);
    setErrors(newErrors);
    return Object.values(newErrors).every((error) => error === null);
  };

  return (
    <div className={`custom-node dark-theme ${data.status === 'RUNNING' ? 'running' : data.status === 'COMPLETED' ? 'completed' : data.status === 'FAILED' ? 'failed' :''}`}>
      <div className="node-header">
        <div className="node-title">{data.blockType || data.title}</div>
        <Button onClick={toggleProperties} className="toggle-button">
          &#9776;
        </Button>
      </div>
      <div className="node-content">
        <div className="input-section">
          {data.inputSchema &&
            Object.entries(data.inputSchema.properties).map(([key, schema]) => (
              <div key={key}>
                <div className="handle-container">
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={key}
                    style={{ background: '#555', borderRadius: '50%' }}
                  />
                  <span className="handle-label">{key}</span>
                </div>
                {renderInputField(key, schema)}
              </div>
            ))}
        </div>
        <div className="output-section">
          {data.outputSchema && generateHandles(data.outputSchema, 'source')}
        </div>
      </div>
      {isPropertiesOpen && (
        <div className="node-properties">
          <h4>Node Output</h4>
          <p>
            <strong>Status:</strong>{' '}
            {typeof data.status === 'object' ? JSON.stringify(data.status) : data.status || 'N/A'}
          </p>
          <p>
            <strong>Output Data:</strong>{' '}
            {typeof data.output_data === 'object'
              ? JSON.stringify(data.output_data)
              : data.output_data || 'N/A'}
          </p>
        </div>
      )}
      <ModalComponent
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        value={modalValue}
        key={activeKey}
      />
    </div>
  );
};

export default memo(CustomNode);
