# Form Creator System

The Form Creator is a dynamic form generation system built on React JSON Schema Form (RJSF) that automatically creates interactive forms based on JSON schemas. It's the core component that powers the input handling in the FlowEditor.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [Schema Processing](#schema-processing)
- [Widget System](#widget-system)
- [Field System](#field-system)
- [Template System](#template-system)
- [Customization Guide](#customization-guide)
- [Advanced Features](#advanced-features)

## Architecture Overview

The Form Creator system consists of several interconnected layers:

```
FormCreator
├── Schema Preprocessing
│   └── input-schema-pre-processor.ts
├── Widget System
│   ├── TextInputWidget
│   ├── SelectWidget
│   ├── SwitchWidget
│   └── ... (other widgets)
├── Field System
│   ├── AnyOfField
│   ├── ObjectField
│   └── CredentialsField
├── Template System
│   ├── FieldTemplate
│   └── ArrayFieldTemplate
└── UI Schema
    └── uiSchema.ts
```

## How It Works

### 1. **Schema Input**

The FormCreator receives a JSON schema that defines the structure of the form:

```typescript
const schema = {
  type: "object",
  properties: {
    message: {
      type: "string",
      title: "Message",
      description: "Enter your message",
    },
    count: {
      type: "number",
      title: "Count",
      minimum: 0,
    },
  },
};
```

### 2. **Schema Preprocessing**

The schema is preprocessed to ensure all properties have proper types:

```typescript
// Before preprocessing
{
  "properties": {
    "name": { "title": "Name" }  // No type defined
  }
}

// After preprocessing
// if there is no type - that means it can accept any type
{
  "properties": {
    "name": {
      "title": "Name",
      "anyOf": [
        { "type": "string" },
        { "type": "number" },
        { "type": "boolean" },
        { "type": "array", "items": { "type": "string" } },
        { "type": "object" },
        { "type": "null" }
      ]
    }
  }
}
```

### 3. **Widget Mapping**

Schema types are mapped to appropriate input widgets:

```typescript
// Schema type -> Widget mapping
"string" -> TextInputWidget
"number" -> TextInputWidget (with number type)
"boolean" -> SwitchWidget
"array" -> ArrayFieldTemplate
"object" -> ObjectField
"enum" -> SelectWidget
```

### 4. **Form Rendering**

RJSF renders the form using the mapped widgets and templates:

```typescript
<Form
  schema={preprocessedSchema}
  validator={validator}
  fields={fields}
  templates={templates}
  widgets={widgets}
  formContext={{ nodeId }}
  onChange={handleChange}
  uiSchema={uiSchema}
/>
```

## Schema Processing

### Input Schema Preprocessor

The `preprocessInputSchema` function ensures all properties have proper types:

```typescript
export function preprocessInputSchema(schema: RJSFSchema): RJSFSchema {
  // Recursively processes properties
  if (processedSchema.properties) {
    for (const [key, property] of Object.entries(processedSchema.properties)) {
      // Add type if none exists
      if (
        !processedProperty.type &&
        !processedProperty.anyOf &&
        !processedProperty.oneOf &&
        !processedProperty.allOf
      ) {
        processedProperty.anyOf = [
          { type: "string" },
          { type: "number" },
          { type: "integer" },
          { type: "boolean" },
          { type: "array", items: { type: "string" } },
          { type: "object" },
          { type: "null" },
        ];
      }
    }
  }
}
```

### Key Features

1. **Type Safety**: Ensures all properties have types
2. **Recursive Processing**: Handles nested objects and arrays
3. **Array Item Processing**: Processes array item schemas
4. **Schema Cleanup**: Removes titles and descriptions from root schema

## Widget System

Widgets are the actual input components that users interact with.

### Available Widgets

#### TextInputWidget

Handles text, number, password, and textarea inputs:

```typescript
export const TextInputWidget = (props: WidgetProps) => {
  const { schema } = props;
  const mapped = mapJsonSchemaTypeToInputType(schema);

  const inputConfig = {
    [InputType.TEXT_AREA]: {
      htmlType: "textarea",
      placeholder: "Enter text...",
      handleChange: (v: string) => (v === "" ? undefined : v),
    },
    [InputType.PASSWORD]: {
      htmlType: "password",
      placeholder: "Enter secret text...",
      handleChange: (v: string) => (v === "" ? undefined : v),
    },
    [InputType.NUMBER]: {
      htmlType: "number",
      placeholder: "Enter number value...",
      handleChange: (v: string) => (v === "" ? undefined : Number(v)),
    }
  };

  return <Input {...config} />;
};
```

#### SelectWidget

Handles dropdown and multi-select inputs:

```typescript
export const SelectWidget = (props: WidgetProps) => {
  const { options, value, onChange, schema } = props;
  const enumOptions = options.enumOptions || [];
  const type = mapJsonSchemaTypeToInputType(schema);

  if (type === InputType.MULTI_SELECT) {
    return <MultiSelector values={value} onValuesChange={onChange} />;
  }

  return <Select value={value} onValueChange={onChange} options={enumOptions} />;
};
```

#### SwitchWidget

Handles boolean toggles:

```typescript
export function SwitchWidget(props: WidgetProps) {
  const { value = false, onChange, disabled, readonly } = props;

  return (
    <Switch
      checked={Boolean(value)}
      onCheckedChange={(checked) => onChange(checked)}
      disabled={disabled || readonly}
    />
  );
}
```

### Widget Registration

Widgets are registered in the widgets registry:

```typescript
export const widgets: RegistryWidgetsType = {
  TextWidget: TextInputWidget,
  SelectWidget: SelectWidget,
  CheckboxWidget: SwitchWidget,
  FileWidget: FileWidget,
  DateWidget: DateInputWidget,
  TimeWidget: TimeInputWidget,
  DateTimeWidget: DateTimeInputWidget,
};
```

## Field System

Fields handle complex data structures and provide custom rendering logic.

### AnyOfField

Handles union types and nullable fields:

```typescript
export const AnyOfField = ({ schema, formData, onChange, ...props }: FieldProps) => {
  const { isNullableType, selectedType, handleTypeChange, currentTypeOption } = useAnyOfField(schema, formData, onChange);

  if (isNullableType) {
    return (
      <div>
        <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
        <Switch checked={isEnabled} onCheckedChange={handleNullableToggle} />
        {isEnabled && renderInput(nonNull)}
      </div>
    );
  }

  return (
    <div>
      <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
      <Select value={selectedType} onValueChange={handleTypeChange} />
      {renderInput(currentTypeOption)}
    </div>
  );
};
```

### ObjectField

Handles free-form object editing:

```typescript
export const ObjectField = (props: FieldProps) => {
  const { schema, formData = {}, onChange, name, idSchema, formContext } = props;

  // Use default field for fixed-schema objects
  if (idSchema?.$id === "root" || !isFreeForm) {
    return <DefaultObjectField {...props} />;
  }

  // Use custom ObjectEditor for free-form objects
  return (
    <ObjectEditor
      id={`${name}-input`}
      nodeId={nodeId}
      fieldKey={fieldKey}
      value={formData}
      onChange={onChange}
    />
  );
};
```

### Field Registration

Fields are registered in the fields registry:

```typescript
export const fields: RegistryFieldsType = {
  AnyOfField: AnyOfField,
  credentials: CredentialsField,
  ObjectField: ObjectField,
};
```

## Template System

Templates provide custom rendering for form structure elements.

### FieldTemplate

Custom field wrapper with connection handles:

```typescript
const FieldTemplate: React.FC<FieldTemplateProps> = ({
  id, label, required, description, children, schema, formContext, uiSchema
}) => {
  const { isInputConnected } = useEdgeStore();
  const { nodeId } = formContext;

  const fieldKey = generateHandleId(id);
  const isConnected = isInputConnected(nodeId, fieldKey);

  return (
    <div className="mt-4 w-[400px] space-y-1">
      {label && schema.type && (
        <label htmlFor={id} className="flex items-center gap-1">
          <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
          <Text variant="body">{label}</Text>
          <Text variant="small" className={colorClass}>({displayType})</Text>
          {required && <span style={{ color: "red" }}>*</span>}
        </label>
      )}
      {!isConnected && <div className="pl-2">{children}</div>}
    </div>
  );
};
```

### ArrayFieldTemplate

Custom array editing interface:

```typescript
function ArrayFieldTemplate(props: ArrayFieldTemplateProps) {
  const { items, canAdd, onAddClick, disabled, readonly, formContext, idSchema } = props;
  const { nodeId } = formContext;

  return (
    <ArrayEditor
      items={items}
      nodeId={nodeId}
      canAdd={canAdd}
      onAddClick={onAddClick}
      disabled={disabled}
      readonly={readonly}
      id={idSchema.$id}
    />
  );
}
```

## Customization Guide

### Adding a Custom Widget

1. **Create the Widget Component**:

```typescript
import { WidgetProps } from "@rjsf/utils";

export const MyCustomWidget = (props: WidgetProps) => {
  const { value, onChange, schema, disabled, readonly } = props;

  return (
    <div>
      <input
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || readonly}
        placeholder={schema.placeholder}
      />
    </div>
  );
};
```

2. **Register the Widget**:

```typescript
// In widgets/index.ts
export const widgets: RegistryWidgetsType = {
  // ... existing widgets
  MyCustomWidget: MyCustomWidget,
};
```

3. **Use in Schema**:

```typescript
const schema = {
  type: "object",
  properties: {
    myField: {
      type: "string",
      "ui:widget": "MyCustomWidget",
    },
  },
};
```

### Adding a Custom Field

1. **Create the Field Component**:

```typescript
import { FieldProps } from "@rjsf/utils";

export const MyCustomField = (props: FieldProps) => {
  const { schema, formData, onChange, name, idSchema, formContext } = props;

  return (
    <div>
      {/* Custom field implementation */}
    </div>
  );
};
```

2. **Register the Field**:

```typescript
// In fields/index.ts
export const fields: RegistryFieldsType = {
  // ... existing fields
  MyCustomField: MyCustomField,
};
```

3. **Use in Schema**:

```typescript
const schema = {
  type: "object",
  properties: {
    myField: {
      type: "string",
      "ui:field": "MyCustomField",
    },
  },
};
```

### Customizing Templates

1. **Create Custom Template**:

```typescript
const MyCustomFieldTemplate: React.FC<FieldTemplateProps> = (props) => {
  return (
    <div className="my-custom-field">
      {/* Custom template implementation */}
    </div>
  );
};
```

2. **Register Template**:

```typescript
// In templates/index.ts
export const templates = {
  FieldTemplate: MyCustomFieldTemplate,
  // ... other templates
};
```

## Advanced Features

### Connection State Management

The Form Creator integrates with the edge store to show/hide input fields based on connection state:

```typescript
const FieldTemplate = ({ id, children, formContext }) => {
  const { isInputConnected } = useEdgeStore();
  const { nodeId } = formContext;

  const fieldKey = generateHandleId(id);
  const isConnected = isInputConnected(nodeId, fieldKey);

  return (
    <div>
      <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
      {!isConnected && children}
    </div>
  );
};
```

### Advanced Mode

Fields can be hidden/shown based on advanced mode:

```typescript
const FieldTemplate = ({ schema, formContext }) => {
  const { nodeId } = formContext;
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] || false
  );

  if (!showAdvanced && schema.advanced === true) {
    return null;
  }

  return <div>{/* field content */}</div>;
};
```

### Array Item Context

Array items have special context for connection handling:

```typescript
const ArrayEditor = ({ items, nodeId }) => {
  return (
    <div>
      {items?.map((element) => {
        const fieldKey = generateHandleId(id, [element.index.toString()], HandleIdType.ARRAY);
        const isConnected = isInputConnected(nodeId, fieldKey);

        return (
          <ArrayEditorContext.Provider
            value={{ isArrayItem: true, fieldKey, isConnected }}
          >
            {element.children}
          </ArrayEditorContext.Provider>
        );
      })}
    </div>
  );
};
```

### Handle ID Generation

Handle IDs are generated based on field structure:

```typescript
// Simple field
generateHandleId("message"); // "message"

// Nested field
generateHandleId("config", ["api_key"]); // "config.api_key"

// Array item
generateHandleId("items", ["0"]); // "items_$_0"

// Key-value pair
generateHandleId("headers", ["Authorization"]); // "headers_#_Authorization"
```
