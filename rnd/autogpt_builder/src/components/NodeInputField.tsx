import { Cross2Icon, PlusIcon } from "@radix-ui/react-icons";
import { beautifyString } from "@/lib/utils";
import { FC, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

type BlockInputFieldProps = {
  keyName: string
  schema: any
  parentKey?: string
  value: string | Array<string> | { [key: string]: string }
  handleInputClick: (key: string) => void
  handleInputChange: (key: string, value: any) => void
  errors: { [key: string]: string | null }
}

const NodeInputField: FC<BlockInputFieldProps> =
  ({ keyName: key, schema, parentKey = '', value, handleInputClick, handleInputChange, errors }) => {
    const fullKey = parentKey ? `${parentKey}.${key}` : key;
    const error = errors[fullKey];
    const displayKey = schema.title || beautifyString(key);

    const [keyValuePairs, _setKeyValuePairs] = useState<{ key: string, value: string }[]>(
      "additionalProperties" in schema && value
        ? Object.entries(value).map(([key, value]) => ({ key: key, value: value}))
        : []
    );

    function setKeyValuePairs(newKVPairs: typeof keyValuePairs): void {
      _setKeyValuePairs(newKVPairs);
      handleInputChange(
        fullKey,
        newKVPairs.reduce((obj, {key, value}) => ({ ...obj, [key]: value }), {})
      );
    }

    const renderClickableInput = (value: string | null = null, placeholder: string = "", secret: boolean = false) => {
      // if secret is true, then the input field will be a password field if the value is not null
      return secret ? (
        <div className="clickable-input" onClick={() => handleInputClick(fullKey)}>
          {value ? <i className="text-gray-500">********</i> : <i className="text-gray-500">{placeholder}</i>}
        </div>
      ) : (
        <div className="clickable-input" onClick={() => handleInputClick(fullKey)}>
          {value || <i className="text-gray-500">{placeholder}</i>}
        </div>
      )
    };

    if (schema.type === 'object' && schema.properties) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {Object.entries(schema.properties).map(([propKey, propSchema]: [string, any]) => (
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

    if (schema.type === 'object' && schema.additionalProperties) {
      return (
        <div key={fullKey}>
          <div>
            {keyValuePairs.map(({ key, value }, index) => (
              <div key={index} className="flex items-center w-[325px] space-x-2 mb-2">
                <Input
                  type="text"
                  placeholder="Key"
                  value={key}
                  onChange={(e) => setKeyValuePairs(
                    keyValuePairs.toSpliced(index, 1, {
                      key: e.target.value, value: value
                    })
                  )}
                />
                <Input
                  type="text"
                  placeholder="Value"
                  value={value}
                  onChange={(e) => setKeyValuePairs(
                    keyValuePairs.toSpliced(index, 1, {
                      key: key, value: e.target.value
                    })
                  )}
                />
                <Button variant="ghost" className="px-2"
                  onClick={() => setKeyValuePairs(keyValuePairs.toSpliced(index, 1))}
                >
                  <Cross2Icon />
                </Button>
              </div>
            ))}
            <Button className="w-full"
              onClick={() => setKeyValuePairs(
                keyValuePairs.concat({ key: "", value: "" })
              )}
            >
              <PlusIcon className="mr-2" /> Add Property
            </Button>
          </div>
          {error && <span className="error-message">{error}</span>}
        </div>
      );
    }

    if (schema.anyOf) {
      const types = schema.anyOf.map((s: any) => s.type);
      if (types.includes('string') && types.includes('null')) {
        return (
          <div key={fullKey} className="input-container">
            {renderClickableInput(value as string, schema.placeholder || `Enter ${displayKey} (optional)`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
      }
    }

    if (schema.allOf) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {schema.allOf[0].properties && Object.entries(schema.allOf[0].properties).map(([propKey, propSchema]: [string, any]) => (
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

    if (schema.oneOf) {
      return (
        <div key={fullKey} className="object-input">
          <strong>{displayKey}:</strong>
          {schema.oneOf[0].properties && Object.entries(schema.oneOf[0].properties).map(([propKey, propSchema]: [string, any]) => (
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
            <input
              type="number"
              value={value as string || ''}
              onChange={(e) => handleInputChange(fullKey, parseFloat(e.target.value))}
              className="number-input"
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
            {renderClickableInput(value as string, schema.placeholder || `Enter ${beautifyString(displayKey)} (Complex)`)}
            {error && <span className="error-message">{error}</span>}
          </div>
        );
    }
  }

export default NodeInputField;
