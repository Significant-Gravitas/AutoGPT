import React, { useState, useEffect, FC, memo, useRef } from 'react';
import { NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';
import './customnode.css';
import InputModalComponent from './InputModalComponent';
import OutputModalComponent from './OutputModalComponent';
import { BlockSchema } from '@/lib/types';
import { beautifyString } from '@/lib/utils';
import { Switch } from "@/components/ui/switch"
import NodeHandle from './NodeHandle';
import NodeInputField from './NodeInputField';

type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: BlockSchema;
  outputSchema: BlockSchema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>;
  isOutputOpen: boolean;
  status?: string;
  output_data?: any;
};

const CustomNode: FC<NodeProps<CustomNodeData>> = ({ data, id }) => {
  const [isOutputOpen, setIsOutputOpen] = useState(data.isOutputOpen || false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [modalValue, setModalValue] = useState<string>('');
  const [errors, setErrors] = useState<{ [key: string]: string | null }>({});
  const [isOutputModalOpen, setIsOutputModalOpen] = useState(false);
  const outputDataRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (data.output_data || data.status) {
      setIsOutputOpen(true);
    }
  }, [data.output_data, data.status]);

  useEffect(() => {
    console.log(`Node ${id} data:`, data);
  }, [id, data]);

  const toggleOutput = (checked: boolean) => {
    setIsOutputOpen(checked);
  };

  const toggleAdvancedSettings = (checked: boolean) => {
    setIsAdvancedOpen(checked);
  };

  const hasOptionalFields = () => {
    return data.inputSchema && Object.keys(data.inputSchema.properties).some((key) => {
      return !(data.inputSchema.required?.includes(key));
    });
  };

  const generateOutputHandles = (schema: BlockSchema) => {
    if (!schema?.properties) return null;
    const keys = Object.keys(schema.properties);
    return keys.map((key) => (
      <div key={key}>
        <NodeHandle keyName={key} isConnected={isHandleConnected(key)} schema={schema.properties[key]} side="right" />
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
    console.log(`Getting value for key: ${key}`);
    const keys = key.split('.');
    return keys.reduce((acc, k) => (acc && acc[k] !== undefined) ? acc[k] : '', data.hardcodedValues);
  };

  const isHandleConnected = (key: string) => {
    return data.connections && data.connections.some((conn: any) => {
      if (typeof conn === 'string') {
        const [source, target] = conn.split(' -> ');
        return (target.includes(key) && target.includes(data.title)) ||
          (source.includes(key) && source.includes(data.title));
      }
      return (conn.target === id && conn.targetHandle === key) ||
        (conn.source === id && conn.sourceHandle === key);
    });
  };

  const handleInputClick = (key: string) => {
    console.log(`Opening modal for key: ${key}`);
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

  const handleOutputClick = () => {
    setIsOutputModalOpen(true);
    setModalValue(typeof data.output_data === 'object' ? JSON.stringify(data.output_data, null, 2) : data.output_data);
  };

  const isTextTruncated = (element: HTMLElement | null): boolean => {
    if (!element) return false;
    return element.scrollHeight > element.clientHeight || element.scrollWidth > element.clientWidth;
  };

  return (
    <div className={`custom-node dark-theme ${data.status === 'RUNNING' ? 'running' : data.status === 'COMPLETED' ? 'completed' : data.status === 'FAILED' ? 'failed' : ''}`}>
      <div className="mb-2">
        <div className="text-lg font-bold">{beautifyString(data.blockType?.replace(/Block$/, '') || data.title)}</div>
      </div>
      <div className="node-content">
        <div>
          {data.inputSchema &&
            Object.entries(data.inputSchema.properties).map(([key, schema]) => {
              const isRequired = data.inputSchema.required?.includes(key);
              return (isRequired || isAdvancedOpen) && (
                <div key={key}>
                  <NodeHandle keyName={key} isConnected={isHandleConnected(key)} schema={schema} side="left" />
                  {isHandleConnected(key) ? <></> :
                  <NodeInputField
                    keyName={key}
                    schema={schema}
                    value={getValue(key)}
                    handleInputClick={handleInputClick}
                    handleInputChange={handleInputChange}
                    errors={errors}
                  />}
                </div>
              );
            })}
        </div>
        <div>
          {data.outputSchema && generateOutputHandles(data.outputSchema)}
        </div>
      </div>
      {isOutputOpen && (
        <div className="node-output" onClick={handleOutputClick}>
          <p>
            <strong>Status:</strong>{' '}
            {typeof data.status === 'object' ? JSON.stringify(data.status) : data.status || 'N/A'}
          </p>
          <p>
            <strong>Output Data:</strong>{' '}
            {(() => {
              const outputText = typeof data.output_data === 'object'
                ? JSON.stringify(data.output_data)
                : data.output_data;
              
              if (!outputText) return 'No output data';
              
              return outputText.length > 100
                ? `${outputText.slice(0, 100)}... Press To Read More`
                : outputText;
            })()}
          </p>
        </div>
      )}
      <div className="flex items-center mt-2.5">
        <Switch onCheckedChange={toggleOutput} className='custom-switch' />
        <span className='m-1 mr-4'>Output</span>
        {hasOptionalFields() && (
          <>
            <Switch onCheckedChange={toggleAdvancedSettings} className='custom-switch' />
            <span className='m-1'>Advanced</span>
          </>
        )}
      </div>
      <InputModalComponent
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        value={modalValue}
        key={activeKey}
      />
      <OutputModalComponent
        isOpen={isOutputModalOpen}
        onClose={() => setIsOutputModalOpen(false)}
        value={modalValue}
      />
    </div>
  );
};

export default memo(CustomNode);
