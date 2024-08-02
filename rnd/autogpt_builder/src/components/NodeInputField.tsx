import { beautifyString } from "@/lib/utils";
import { BlockIOSchema } from "@/lib/autogpt-server-api/types";
import { FC, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

type BlockInputFieldProps = {
  keyName: string
  schema: BlockIOSchema
  parentKey?: string
  value: string | Array<string> | { [key: string]: string }
  handleInputClick: (key: string) => void
  handleInputChange: (key: string, value: any) => void
  errors?: { [key: string]: string } | string | null
}

const NodeInputField: FC<BlockInputFieldProps> =
  ({ keyName: key, schema, parentKey = '', value, handleInputClick, handleInputChange, errors }) => {
    const [newKey, setNewKey] = useState<string>('');
    const [newValue, setNewValue] = useState<string>('');
    const [keyValuePairs, setKeyValuePairs] = useState<{ key: string, value: string }[]>([]);

    const fullKey = parentKey ? `${parentKey}.${key}` : key;
    const error = typeof errors === 'string' ? errors : errors?.[key] ?? "";
    const displayKey = schema.title || beautifyString(key);

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

    const renderClickableInput = (value: string | null = null, placeholder: string = "", secret: boolean = false) => {
      const className = `clickable-input ${error ? 'border-error' : ''}`

      // if secret is true, then the input field will be a password field if the value is not null
      return secret ? (
        <div className={className} onClick={() => handleInputClick(fullKey)}>
          {value ? <span>********</span> : <i className="text-gray-500">{placeholder}</i>}
        </div>
      ) : (
        <div className={className} onClick={() => handleInputClick(fullKey)}>
          {value || <i className="text-gray-500">{placeholder}</i>}
        </div>
      )
    };

    if ("properties" in schema) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {Object.entries(schema.properties).map(([propKey, propSchema]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              <NodeInputField
                keyName={propKey}
                schema={propSchema}
                parentKey={fullKey}
                value={(value as { [key: string]: string })[propKey]}
                handleInputClick={handleInputClick}
                handleInputChange={handleInputChange}
                errors={errors}
              />
            </div>
          ))}
        </div>
      );
    }

    if ("additionalProperties" in schema) {
      const objectValue = value || {};
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {Object.entries(objectValue).map(([propKey, propValue]: [string, any]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              <div className="clickable-input" onClick={() => handleInputClick(`${fullKey}.${propKey}`)}>
                {beautifyString(propKey)}: {typeof propValue === 'object' ? JSON.stringify(propValue, null, 2) : propValue}
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
                    value={beautifyString(pair.key)}
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
                    value={beautifyString(pair.value)}
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

    if ("anyOf" in schema) {
      const types = schema.anyOf.map(s => "type" in s ? s.type : undefined);
      if (types.includes('string') && types.includes('null')) {
        return (
          <div key={fullKey} className="input-container">
            {renderClickableInput(value as string, schema.placeholder || `Enter ${displayKey} (optional)`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      }
    }

    if ("allOf" in schema) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {"properties" in schema.allOf[0] &&
          Object.entries(schema.allOf[0].properties).map(([propKey, propSchema]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              <NodeInputField
                keyName={propKey}
                schema={propSchema}
                parentKey={fullKey}
                value={(value as { [key: string]: string })[propKey]}
                handleInputClick={handleInputClick}
                handleInputChange={handleInputChange}
                errors={errors}
              />
            </div>
          ))}
        </div>
      );
    }

    if ("oneOf" in schema) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {"properties" in schema.oneOf[0] &&
          Object.entries(schema.oneOf[0].properties).map(([propKey, propSchema]) => (
            <div key={`${fullKey}.${propKey}`} className="nested-input">
              <NodeInputField
                keyName={propKey}
                schema={propSchema}
                parentKey={fullKey}
                value={(value as { [key: string]: string })[propKey]}
                handleInputClick={handleInputClick}
                handleInputChange={handleInputChange}
                errors={errors}
              />
            </div>
          ))}
        </div>
      );
    }

    if (!("type" in schema)) {
      console.warn(`Schema for input ${key} does not specify a type:`, schema);
      return (
        <div key={fullKey} className="input-container">
          {renderClickableInput(value as string, schema.placeholder || `Enter ${beautifyString(displayKey)} (Complex)`)}
          {error && <span className="error-message">{error}</span>}
        </div>
      );
    }

    switch (schema.type) {
      case 'string':
        if (schema.enum) {

          return (
            <div key={fullKey} className="input-container">
              <select
                value={value as string || ''}
                onChange={(e) => handleInputChange(fullKey, e.target.value)}
                className="select-input"
              >
                <option value="">Select {displayKey}</option>
                {schema.enum.map((option: string) => (
                  <option key={option} value={option}>
                    {beautifyString(option)}
                  </option>
                ))}
              </select>
              {error && <span className="error-message">{error}</span>}
            </div>
          )
        }

        else if (schema.secret) {
          return (<div key={fullKey} className="input-container">
            {renderClickableInput(value as string, schema.placeholder || `Enter ${displayKey}`, true)}
            {error && <span className="error-message">{error}</span>}
          </div>)

        }
        else {
          return (
            <div key={fullKey} className="input-container">
              {renderClickableInput(value as string, schema.placeholder || `Enter ${displayKey}`)}
              {error && <span className="error-message">{error}</span>}
            </div>
          );
        }
      case 'boolean':
        return (
          <div key={fullKey} className="input-container">
            <select
              value={value === undefined ? '' : value.toString()}
              onChange={(e) => handleInputChange(fullKey, e.target.value === 'true')}
              className="select-input"
            >
              <option value="">Select {displayKey}</option>
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
            <Input
              type="number"
              value={value as string || ''}
              onChange={(e) => handleInputChange(fullKey, parseFloat(e.target.value))}
              className={`number-input ${error ? 'border-error' : ''}`}
            />
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      case 'array':
        if (schema.items && schema.items.type === 'string') {
          const arrayValues = value as Array<string> || [];
          return (
            <div key={fullKey} className="input-container">
              {arrayValues.map((item: string, index: number) => (
                <div key={`${fullKey}.${index}`} className="array-item-container">
                  <Input
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
              {error && <span className="error-message ml-2">{error}</span>}
            </div>
          );
        }
        return null;
      default:
        console.warn(`Schema for input ${key} specifies unknown type:`, schema);
        return (
          <div key={fullKey} className="input-container">
            {renderClickableInput(value as string, schema.placeholder || `Enter ${beautifyString(displayKey)} (Complex)`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
    }
  }

export default NodeInputField;
