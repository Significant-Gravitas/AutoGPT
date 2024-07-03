import React, { useState, useEffect, FC, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';
import './customnode.css';
import ModalComponent from './ModalComponent';

type Schema = {
  type: string;
  properties: { [key: string]: any };
  required?: string[];
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
    const newValues = { ...data.hardcodedValues, [key]: value };
    console.log(`Updating hardcoded values for node ${id}:`, newValues);
    data.setHardcodedValues(newValues);
    setErrors((prevErrors) => ({ ...prevErrors, [key]: null }));
  };

  const validateInput = (key: string, value: any, schema: any) => {
    switch (schema.type) {
      case 'string':
        if (schema.enum && !schema.enum.includes(value)) {
          return `Invalid value for ${key}`;
        }
        break;
      case 'boolean':
        if (typeof value !== 'boolean') {
          return `Invalid value for ${key}`;
        }
        break;
      case 'number':
        if (typeof value !== 'number') {
          return `Invalid value for ${key}`;
        }
        break;
      case 'array':
        if (!Array.isArray(value) || value.some((item: any) => typeof item !== 'string')) {
          return `Invalid value for ${key}`;
        }
        if (schema.minItems && value.length < schema.minItems) {
          return `${key} requires at least ${schema.minItems} items`;
        }
        break;
      default:
        return null;
    }
    return null;
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

  const handleInputClick = (key: string) => {
    setActiveKey(key);
    setModalValue(data.hardcodedValues[key] || '');
    setIsModalOpen(true);
  };

  const handleModalSave = (value: string) => {
    if (activeKey) {
      handleInputChange(activeKey, value);
    }
    setIsModalOpen(false);
    setActiveKey(null);
  };

  const addArrayItem = (key: string) => {
    const currentValues = data.hardcodedValues[key] || [];
    handleInputChange(key, [...currentValues, '']);
  };

  const removeArrayItem = (key: string, index: number) => {
    const currentValues = data.hardcodedValues[key] || [];
    currentValues.splice(index, 1);
    handleInputChange(key, [...currentValues]);
  };

  const handleArrayItemChange = (key: string, index: number, value: string) => {
    const currentValues = data.hardcodedValues[key] || [];
    currentValues[index] = value;
    handleInputChange(key, [...currentValues]);
  };

  const addDynamicTextInput = () => {
    const dynamicKeyPrefix = 'texts_$_';
    const currentKeys = Object.keys(data.hardcodedValues).filter(key => key.startsWith(dynamicKeyPrefix));
    const nextIndex = currentKeys.length + 1;
    const newKey = `${dynamicKeyPrefix}${nextIndex}`;
    handleInputChange(newKey, '');
  };

  const removeDynamicTextInput = (key: string) => {
    const newValues = { ...data.hardcodedValues };
    delete newValues[key];
    data.setHardcodedValues(newValues);
  };

  const handleDynamicTextInputChange = (key: string, value: string) => {
    handleInputChange(key, value);
  };

  const renderInputField = (key: string, schema: any) => {
    const error = errors[key];
    switch (schema.type) {
      case 'string':
        return schema.enum ? (
          <div key={key} className="input-container">
            <select
              value={data.hardcodedValues[key] || ''}
              onChange={(e) => handleInputChange(key, e.target.value)}
              className="select-input"
            >
              {schema.enum.map((option: string) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {error && <span className="error-message">{error}</span>}
          </div>
        ) : (
          <div key={key} className="input-container">
            <div className="clickable-input" onClick={() => handleInputClick(key)}>
              {data.hardcodedValues[key] || `Enter ${key}`}
            </div>
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'boolean':
        return (
          <div key={key} className="input-container">
            <label className="radio-label">
              <input
                type="radio"
                value="true"
                checked={data.hardcodedValues[key] === true}
                onChange={() => handleInputChange(key, true)}
              />
              True
            </label>
            <label className="radio-label">
              <input
                type="radio"
                value="false"
                checked={data.hardcodedValues[key] === false}
                onChange={() => handleInputChange(key, false)}
              />
              False
            </label>
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'number':
        return (
          <div key={key} className="input-container">
            <input
              type="number"
              value={data.hardcodedValues[key] || ''}
              onChange={(e) => handleInputChange(key, parseFloat(e.target.value))}
              className="number-input"
            />
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'array':
        if (schema.items && schema.items.type === 'string') {
          const arrayValues = data.hardcodedValues[key] || [];
          return (
            <div key={key} className="input-container">
              {arrayValues.map((item: string, index: number) => (
                <div key={`${key}-${index}`} className="array-item-container">
                  <input
                    type="text"
                    value={item}
                    onChange={(e) => handleArrayItemChange(key, index, e.target.value)}
                    className="array-item-input"
                  />
                  <button onClick={() => removeArrayItem(key, index)} className="array-item-remove">
                    &times;
                  </button>
                </div>
              ))}
              <button onClick={() => addArrayItem(key)} className="array-item-add">
                Add Item
              </button>
              {error && <span className="error-message">{error}</span>}
            </div>
          );
        }
        return null;
      default:
        return null;
    }
  };

  const renderDynamicTextFields = () => {
    const dynamicKeyPrefix = 'texts_$_';
    const dynamicKeys = Object.keys(data.hardcodedValues).filter(key => key.startsWith(dynamicKeyPrefix));

    return dynamicKeys.map((key, index) => (
      <div key={key} className="input-container">
        <div className="handle-container">
          <Handle
            type="target"
            position={Position.Left}
            id={key}
            style={{ background: '#555', borderRadius: '50%' }}
          />
          <span className="handle-label">{key}</span>
          {!isHandleConnected(key) && (
            <>
              <input
                type="text"
                value={data.hardcodedValues[key]}
                onChange={(e) => handleDynamicTextInputChange(key, e.target.value)}
                className="dynamic-text-input"
              />
              <button onClick={() => removeDynamicTextInput(key)} className="array-item-remove">
                &times;
              </button>
            </>
          )}
        </div>
      </div>
    ));
  };

  const validateInputs = () => {
    const newErrors: { [key: string]: string | null } = {};
    Object.keys(data.inputSchema.properties).forEach((key) => {
      const value = data.hardcodedValues[key];
      const schema = data.inputSchema.properties[key];
      const error = validateInput(key, value, schema);
      if (error) {
        newErrors[key] = error;
      }
    });
    setErrors(newErrors);
    return Object.values(newErrors).every((error) => error === null);
  };

  const handleSubmit = () => {
    if (validateInputs()) {
      console.log("Valid data:", data.hardcodedValues);
    } else {
      console.log("Invalid data:", errors);
    }
  };


  return (
    <div className="custom-node">
      <div className="node-header">
        <div className="node-title">{data.blockType || data.title}</div>
        <button onClick={toggleProperties} className="toggle-button">
          &#9776;
        </button>
      </div>
      <div className="node-content">
        <div className="input-section">
          {data.inputSchema &&
            Object.keys(data.inputSchema.properties).map((key) => (
              <div key={key}>
                {key !== 'texts' ? (
                  <div>
                    <div className="handle-container">
                      <Handle
                        type="target"
                        position={Position.Left}
                        id={key}
                        style={{ background: '#555', borderRadius: '50%' }}
                      />
                      <span className="handle-label">{key}</span>
                    </div>
                    {!isHandleConnected(key) && renderInputField(key, data.inputSchema.properties[key])}
                  </div>
                ) : (
                  <div key={key} className="input-container">
                    <div className="handle-container">
                      <Handle
                        type="target"
                        position={Position.Left}
                        id={key}
                        style={{ background: '#555', borderRadius: '50%' }}
                      />
                      <span className="handle-label">{key}</span>
                    </div>
                    {renderDynamicTextFields()}
                    <button onClick={addDynamicTextInput} className="array-item-add">
                      Add Text Input
                    </button>
                  </div>
                )}
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
      <button onClick={handleSubmit}>Submit</button>
      <ModalComponent
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        value={modalValue}
      />
    </div>
  );
};

export default memo(CustomNode);