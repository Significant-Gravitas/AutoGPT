import React from 'react';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { beautifyString } from '@/lib/utils';
import { InputFieldProps } from './types';

export const InputField: React.FC<InputFieldProps> = ({
  schema,
  fullKey,
  displayKey,
  value,
  error,
  handleInputChange,
  handleInputClick,
}) => {
  const renderClickableInput = (value: string | null = null, placeholder: string = "") => (
    <div className="clickable-input" onClick={() => handleInputClick(fullKey)}>
      {value || <i className="text-gray-500">{placeholder}</i>}
    </div>
  );

  if (schema.type === 'object' && schema.properties) {
    return (
      <div className="object-input">
        <strong>{displayKey}:</strong>
        {Object.entries(schema.properties).map(([propKey, propSchema]: [string, any]) => (
          <div key={`${fullKey}.${propKey}`} className="nested-input">
            <InputField
              schema={propSchema}
              fullKey={`${fullKey}.${propKey}`}
              displayKey={propSchema.title || beautifyString(propKey)}
              value={value && value[propKey]}
              error={error}
              handleInputChange={handleInputChange}
              handleInputClick={handleInputClick}
            />
          </div>
        ))}
      </div>
    );
  }

  if (schema.type === 'string') {
    return schema.enum ? (
      <div className="input-container">
        <select
          value={value || ''}
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
    ) : (
      <div className="input-container">
        {renderClickableInput(value, schema.placeholder || `Enter ${displayKey}`)}
        {error && <span className="error-message">{error}</span>}
      </div>
    );
  }

  if (schema.type === 'boolean') {
    return (
      <div className="input-container">
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
  }

  if (schema.type === 'number' || schema.type === 'integer') {
    return (
      <div className="input-container">
        <Input
          type="number"
          value={value || ''}
          onChange={(e) => handleInputChange(fullKey, parseFloat(e.target.value))}
          className="number-input"
        />
        {error && <span className="error-message">{error}</span>}
      </div>
    );
  }

  // Default case for complex or unhandled types
  return (
    <div className="input-container">
      {renderClickableInput(value, schema.placeholder || `Enter ${beautifyString(displayKey)} (Complex)`)}
      {error && <span className="error-message">{error}</span>}
    </div>
  );
};