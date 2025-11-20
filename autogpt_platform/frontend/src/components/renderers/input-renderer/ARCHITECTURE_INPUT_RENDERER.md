# Input-Renderer Architecture Documentation

## Overview

The Input-Renderer is a **JSON Schema-based form generation system** built on top of **React JSON Schema Form (RJSF)**. It dynamically creates form inputs for block nodes in the FlowEditor based on JSON schemas defined in the backend.

This system allows blocks to define their input requirements declaratively, and the frontend automatically generates appropriate UI components.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FormRenderer                         │
│         (Entry point, wraps RJSF Form)                 │
└─────────────────────┬───────────────────────────────────┘
                      │
            ┌─────────▼─────────┐
            │   RJSF Core       │
            │   <Form />        │
            └───────┬───────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        │           │           │              │
   ┌────▼────┐ ┌───▼────┐ ┌────▼─────┐  ┌────▼────┐
   │ Fields  │ │Templates│ │ Widgets  │  │ Schemas │
   └─────────┘ └─────────┘ └──────────┘  └─────────┘
        │           │           │              │
        │           │           │              │
    Handles     Wrapper      Actual       JSON Schema
    complex     layouts      input       (from backend)
    types       & labels    components
```

---

## What is RJSF (React JSON Schema Form)?

**RJSF** is a library that generates React forms from JSON Schema definitions. It follows a specific hierarchy to render forms:

### **RJSF Rendering Flow:**

```
1. JSON Schema (defines data structure)
   ↓
2. Schema Field (decides which Field component to use)
   ↓
3. Field Component (handles specific type logic)
   ↓
4. Field Template (wraps field with label, description)
   ↓
5. Widget (actual input element - TextInput, Select, etc.)
```

### **Example Flow:**

```json
// JSON Schema
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "title": "Name"
    }
  }
}
```

**Becomes:**

```
SchemaField (detects "string" type)
   ↓
StringField (default RJSF field)
   ↓
FieldTemplate (adds label "Name")
   ↓
TextWidget (renders <input type="text" />)
```

---

## Core Components of Input-Renderer

### 1. **FormRenderer** (`FormRenderer.tsx`)

The main entry point that wraps RJSF `<Form />` component.

```typescript
export const FormRenderer = ({
  jsonSchema,        // JSON Schema from backend
  handleChange,      // Callback when form changes
  uiSchema,          // UI customization
  initialValues,     // Pre-filled values
  formContext,       // Extra context (nodeId, uiType, etc.)
}: FormRendererProps) => {
  const preprocessedSchema = preprocessInputSchema(jsonSchema);

  return (
    <Form
      schema={preprocessedSchema}      // Modified schema
      validator={customValidator}      // Custom validation logic
      fields={fields}                  // Custom field components
      templates={templates}            // Custom layout templates
      widgets={widgets}                // Custom input widgets
      formContext={formContext}        // Pass context down
      onChange={handleChange}          // Form change handler
      uiSchema={uiSchema}             // UI customization
      formData={initialValues}        // Initial values
    />
  );
};
```

**Key Props:**

- **`fields`** - Custom components for complex types (anyOf, credentials, objects)
- **`templates`** - Layout wrappers (FieldTemplate, ArrayFieldTemplate)
- **`widgets`** - Actual input components (TextInput, Select, FileWidget)
- **`formContext`** - Shared data (nodeId, showHandles, size)

---

### 2. **Schema Pre-Processing** (`utils/input-schema-pre-processor.ts`)

Before rendering, schemas are transformed to ensure RJSF compatibility.

**Purpose:**

- Add missing `type` fields (prevents RJSF errors)
- Recursively process nested objects and arrays
- Normalize inconsistent schemas from backend

**Example:**

```typescript
// Backend schema (missing type)
{
  "properties": {
    "value": {}  // No type defined!
  }
}

// After pre-processing
{
  "properties": {
    "value": {
      "anyOf": [
        { "type": "string" },
        { "type": "number" },
        { "type": "boolean" },
        // ... all possible types
      ]
    }
  }
}
```

**Why?** RJSF requires explicit types. Without this, it would crash or render incorrectly.

---

## The Three Pillars: Fields, Templates, Widgets

### **A. Fields** (`fields/`)

Fields handle **complex type logic** that goes beyond simple inputs.

**Registered Fields:**

```typescript
export const fields: RegistryFieldsType = {
  AnyOfField: AnyOfField, // Handles anyOf/oneOf
  credentials: CredentialsField, // OAuth/API key handling
  ObjectField: ObjectField, // Free-form objects
};
```

#### **1. AnyOfField** (`fields/AnyOfField/AnyOfField.tsx`)

Handles schemas with multiple possible types (union types).

**When Used:**

```json
{
  "anyOf": [{ "type": "string" }, { "type": "number" }, { "type": "boolean" }]
}
```

**Rendering:**

```
┌─────────────────────────────────────┐
│ Parameter Name (string) ▼           │ ← Type selector dropdown
├─────────────────────────────────────┤
│ [Text Input]                        │ ← Widget for selected type
└─────────────────────────────────────┘
```

**Features:**

- Type selector dropdown
- Nullable types (with toggle switch)
- Recursive rendering (can contain arrays, objects)
- Connection-aware (hides input when connected)

**Special Case: Nullable Types**

```json
{
  "anyOf": [{ "type": "string" }, { "type": "null" }]
}
```

**Renders as:**

```
┌─────────────────────────────────────┐
│ Parameter Name (string | null) [✓]  │ ← Toggle switch
├─────────────────────────────────────┤
│ [Text Input] (only if enabled)      │
└─────────────────────────────────────┘
```

---

#### **2. CredentialsField** (`fields/CredentialField/CredentialField.tsx`)

Handles authentication credentials (OAuth, API Keys, Passwords).

**When Used:**

```json
{
  "type": "object",
  "credentials": {
    "provider": "google",
    "scopes": ["email", "profile"]
  }
}
```

**Flow:**

```
1. Renders SelectCredential dropdown
   ↓
2. User selects existing credential OR clicks "Add New"
   ↓
3. Modal opens (OAuthModal/APIKeyModal/PasswordModal)
   ↓
4. User authorizes/enters credentials
   ↓
5. Credential saved to backend
   ↓
6. Dropdown shows selected credential
```

**Credential Types:**

- **OAuth** - 3rd party authorization (Google, GitHub, etc.)
- **API Key** - Simple key-based auth
- **Password** - Username/password pairs

---

#### **3. ObjectField** (`fields/ObjectField.tsx`)

Handles free-form objects (key-value pairs).

**When Used:**

```json
{
  "type": "object",
  "additionalProperties": true // Free-form
}
```

vs

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" } // Fixed schema
  }
}
```

**Behavior:**

- **Fixed schema** → Uses default RJSF rendering
- **Free-form** → Uses ObjectEditorWidget (JSON editor)

---

### **B. Templates** (`templates/`)

Templates control **layout and wrapping** of fields.

#### **1. FieldTemplate** (`templates/FieldTemplate.tsx`)

Wraps every field with label, type indicator, and connection handle.

**Rendering Structure:**

```
┌────────────────────────────────────────┐
│ ○ Label (type) ⓘ                      │ ← Handle + Label + Type + Info icon
├────────────────────────────────────────┤
│   [Actual Input Widget]                │ ← The input itself
└────────────────────────────────────────┘
```

**Responsibilities:**

- Shows/hides input based on connection status
- Renders connection handle (NodeHandle)
- Displays type information
- Shows tooltip with description
- Handles "advanced" field visibility
- Formats credential field labels

**Key Logic:**

```typescript
// Hide input if connected
{(isAnyOf || !isConnected) && (
  <div>{children}</div>
)}

// Show handle for most fields
{shouldShowHandle && (
  <NodeHandle handleId={handleId} isConnected={isConnected} />
)}
```

**Context-Aware Behavior:**

- Inside `AnyOfField` → No handle (parent handles it)
- Credential field → Special label formatting
- Array item → Uses parent handle
- INPUT/OUTPUT/WEBHOOK blocks → Different handle positioning

---

#### **2. ArrayFieldTemplate** (`templates/ArrayFieldTemplate.tsx`)

Wraps array fields to use custom ArrayEditorWidget.

**Simple Wrapper:**

```typescript
function ArrayFieldTemplate(props: ArrayFieldTemplateProps) {
  const { items, canAdd, onAddClick, nodeId } = props;

  return (
    <ArrayEditorWidget
      items={items}
      nodeId={nodeId}
      canAdd={canAdd}
      onAddClick={onAddClick}
    />
  );
}
```

---

### **C. Widgets** (`widgets/`)

Widgets are **actual input components** - the final rendered HTML elements.

**Registered Widgets:**

```typescript
export const widgets: RegistryWidgetsType = {
  TextWidget: TextInputWidget, // <input type="text" />
  SelectWidget: SelectWidget, // <select> dropdown
  CheckboxWidget: SwitchWidget, // <Switch> toggle
  FileWidget: FileWidget, // File upload
  DateWidget: DateInputWidget, // Date picker
  TimeWidget: TimeInputWidget, // Time picker
  DateTimeWidget: DateTimeInputWidget, // Combined date+time
};
```

#### **Widget Selection Logic (RJSF)**

RJSF automatically picks the right widget based on schema:

```json
{ "type": "string" }                    → TextWidget
{ "type": "string", "enum": [...] }     → SelectWidget
{ "type": "boolean" }                   → CheckboxWidget
{ "type": "string", "format": "date" }  → DateWidget
{ "type": "string", "format": "time" }  → TimeWidget
```

#### **Special Widgets:**

**1. ArrayEditorWidget** (`widgets/ArrayEditorWidget/`)

```
┌─────────────────────────────────────┐
│ ○ Item 1 [Text Input]  [X Remove]  │
│ ○ Item 2 [Text Input]  [X Remove]  │
│ [+ Add Item]                        │
└─────────────────────────────────────┘
```

**Features:**

- Each array item gets its own connection handle
- Remove button per item
- Add button at bottom
- Context provider for handle management

**ArrayEditorContext:**

```typescript
{
  isArrayItem: true,
  arrayFieldHandleId: "input-items-0", // Unique per item
  isConnected: false
}
```

**2. ObjectEditorWidget** (`widgets/ObjectEditorWidget/`)

- JSON editor for free-form objects
- Key-value pair management
- Used by ObjectField for `additionalProperties: true`

---

## The Complete Rendering Flow

### **Example: Rendering a Text Input**

```json
// Backend Schema
{
  "type": "object",
  "properties": {
    "message": {
      "type": "string",
      "title": "Message",
      "description": "Enter your message"
    }
  }
}
```

**Step-by-Step:**

```
1. FormRenderer receives schema
   ↓
2. preprocessInputSchema() normalizes it
   ↓
3. RJSF <Form /> starts rendering
   ↓
4. SchemaField detects "string" type
   ↓
5. Uses default StringField
   ↓
6. FieldTemplate wraps it:
   - Adds NodeHandle (connection point)
   - Adds label "Message (string)"
   - Adds info icon with description
   ↓
7. TextWidget renders <input />
   ↓
8. User types "Hello"
   ↓
9. onChange callback fires
   ↓
10. FormCreator updates nodeStore.updateNodeData()
```

---

### **Example: Rendering AnyOf Field**

```json
// Backend Schema
{
  "anyOf": [{ "type": "string" }, { "type": "number" }],
  "title": "Value"
}
```

**Rendering Flow:**

```
1. RJSF detects "anyOf"
   ↓
2. Uses AnyOfField (custom field)
   ↓
3. AnyOfField renders:
   ┌─────────────────────────────────┐
   │ ○ Value (string) ▼              │ ← Self-managed handle & selector
   ├─────────────────────────────────┤
   │   [Text Input]                  │ ← Recursively renders SchemaField
   └─────────────────────────────────┘
   ↓
4. User changes type to "number"
   ↓
5. AnyOfField re-renders with NumberWidget
   ↓
6. User enters "42"
   ↓
7. onChange({ type: "number", value: 42 })
```

**Key Point:** AnyOfField **does NOT use FieldTemplate** for itself. It manages its own handle and label to avoid duplication. But it **recursively calls SchemaField** for the selected type, which may use FieldTemplate.

---

### **Example: Rendering Array Field**

```json
// Backend Schema
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "title": "Tags"
}
```

**Rendering Flow:**

```
1. RJSF detects "array" type
   ↓
2. Uses ArrayFieldTemplate
   ↓
3. ArrayFieldTemplate passes to ArrayEditorWidget
   ↓
4. ArrayEditorWidget renders:
   ┌─────────────────────────────────┐
   │ ○ Tag 1 [Text Input] [X]        │ ← Each item wrapped in context
   │ ○ Tag 2 [Text Input] [X]        │
   │ [+ Add Item]                    │
   └─────────────────────────────────┘
   ↓
5. Each item wrapped in ArrayEditorContext
   ↓
6. FieldTemplate reads context:
   - isArrayItem = true
   - Uses arrayFieldHandleId instead of own handle
   ↓
7. TextWidget renders for each item
```

---

## Hierarchy: What Comes First?

This is the **order of execution** from schema to rendered input:

```
1. JSON Schema (from backend)
   ↓
2. preprocessInputSchema() (normalization)
   ↓
3. RJSF <Form /> (library entry point)
   ↓
4. SchemaField (RJSF internal - decides which field)
   ↓
5. Field Component (AnyOfField, CredentialsField, or default)
   ↓
6. Template (FieldTemplate or ArrayFieldTemplate)
   ↓
7. Widget (TextWidget, SelectWidget, etc.)
   ↓
8. Actual HTML (<input>, <select>, etc.)
```

---

## Key Concepts Explained

### **1. Why Custom Fields?**

RJSF's default fields don't handle:

- **AnyOf** - Type selection + dynamic widget switching
- **Credentials** - OAuth flows, modal management
- **Free-form Objects** - JSON editor instead of fixed fields

Custom fields fill these gaps.

---

### **2. Why Templates?**

Templates add **FlowEditor-specific UI**:

- Connection handles (left side dots)
- Type indicators
- Tooltips
- Advanced field hiding
- Connection-aware rendering

Default RJSF templates don't support these features.

---

### **3. Why Custom Widgets?**

Custom widgets provide:

- Consistent styling with design system
- Integration with Zustand stores
- Custom behaviors (e.g., FileWidget uploads)
- Better UX (e.g., SwitchWidget vs checkbox)

---

### **4. FormContext - The Shared State**

FormContext passes data down the RJSF tree:

```typescript
type FormContextType = {
  nodeId?: string; // Which node this form belongs to
  uiType?: BlockUIType; // Block type (INPUT, OUTPUT, etc.)
  showHandles?: boolean; // Show connection handles?
  size?: "small" | "large"; // Form size variant
};
```

**Why?** RJSF components don't have direct access to React props from parent. FormContext provides a channel.

**Usage:**

```typescript
// In FieldTemplate
const { nodeId, showHandles, size } = formContext;

// Check if input is connected
const isConnected = useEdgeStore().isInputConnected(nodeId, handleId);

// Hide input if connected
{!isConnected && <div>{children}</div>}
```

---

### **5. Handle Management**

Connection handles are the **left-side dots** on nodes where edges connect.

**Handle ID Format:**

```typescript
// Regular field
generateHandleId("root_message") → "input-message"

// Array item
generateHandleId("root_tags", ["0"]) → "input-tags-0"
generateHandleId("root_tags", ["1"]) → "input-tags-1"

// Nested field
generateHandleId("root_config_api_key") → "input-config-api_key"
```

**Context Provider Pattern (Arrays):**

```typescript
// ArrayEditorWidget wraps each item
<ArrayEditorContext.Provider
  value={{
    isArrayItem: true,
    arrayFieldHandleId: "input-tags-0"
  }}
>
  {element.children}  // ← FieldTemplate renders here
</ArrayEditorContext.Provider>

// FieldTemplate reads context
const { isArrayItem, arrayFieldHandleId } = useContext(ArrayEditorContext);

// Use array handle instead of generating own
const handleId = isArrayItem ? arrayFieldHandleId : generateHandleId(fieldId);
```

---

## Connection-Aware Rendering

One of the most important features: **hiding inputs when connected**.

**Flow:**

```
1. User connects edge to input handle
   ↓
2. edgeStore.addEdge() creates connection
   ↓
3. Next render cycle:
   - FieldTemplate calls isInputConnected(nodeId, handleId)
   - Returns true
   ↓
4. FieldTemplate hides input:
   {!isConnected && <div>{children}</div>}
   ↓
5. Only handle visible (with blue highlight)
```

**Why?** When a value comes from another node's output, manual input is disabled. The connection provides the value.

**Exception:** AnyOf fields still show type selector when connected (but hide the input).

---

## Advanced Features

### **1. Advanced Field Toggle**

Some fields marked as `advanced: true` in schema are hidden by default.

**Logic in FieldTemplate:**

```typescript
const showAdvanced = useNodeStore((state) => state.nodeAdvancedStates[nodeId]);

if (!showAdvanced && schema.advanced === true && !isConnected) {
  return null; // Hide field
}
```

**UI:** NodeAdvancedToggle button in CustomNode shows/hides these fields.

---

### **2. Nullable Type Handling**

```json
{
  "anyOf": [{ "type": "string" }, { "type": "null" }]
}
```

**AnyOfField detects this pattern and renders:**

```
┌─────────────────────────────────────┐
│ Parameter (string | null)  [✓]      │ ← Switch to enable/disable
├─────────────────────────────────────┤
│ [Input only if enabled]             │
└─────────────────────────────────────┘
```

**State Management:**

```typescript
const [isEnabled, setIsEnabled] = useState(formData !== null);

const handleNullableToggle = (checked: boolean) => {
  setIsEnabled(checked);
  onChange(checked ? "" : null); // Send null when disabled
};
```

---

### **3. Recursive Schema Rendering**

AnyOfField, ObjectField, and ArrayField all recursively call `SchemaField`:

```typescript
const SchemaField = registry.fields.SchemaField;

<SchemaField
  schema={nestedSchema}
  formData={formData}
  onChange={handleChange}
  // ... propagate all props
/>
```

This allows **infinite nesting**: arrays of objects, objects with anyOf fields, etc.

---

## Common Patterns

### **Adding a New Widget**

1. Create widget component in `widgets/`:

```typescript
export const MyWidget = ({ value, onChange, ...props }: WidgetProps) => {
  return <input value={value} onChange={(e) => onChange(e.target.value)} />;
};
```

2. Register in `widgets/index.ts`:

```typescript
export const widgets: RegistryWidgetsType = {
  // ...
  MyCustomWidget: MyWidget,
};
```

3. Use in uiSchema or schema format:

```json
{
  "type": "string",
  "format": "my-custom-format" // RJSF maps format → widget
}
```

---

### **Adding a New Field**

1. Create field component in `fields/`:

```typescript
export const MyField = ({ schema, formData, onChange, ...props }: FieldProps) => {
  // Custom logic here
  return <div>...</div>;
};
```

2. Register in `fields/index.ts`:

```typescript
export const fields: RegistryFieldsType = {
  // ...
  MyField: MyField,
};
```

3. RJSF uses it based on schema structure (e.g., custom keyword).

---

## Integration with FlowEditor

```
CustomNode
    ↓
FormCreator
    ↓
FormRenderer  ← YOU ARE HERE
    ↓
RJSF <Form />
    ↓
(Fields, Templates, Widgets)
    ↓
User Input
    ↓
onChange callback
    ↓
FormCreator.handleChange()
    ↓
nodeStore.updateNodeData(nodeId, { hardcodedValues })
    ↓
historyStore.pushState() (undo/redo)
```

---

## Debugging Tips

### **Field Not Rendering**

- Check if `preprocessInputSchema()` is handling it correctly
- Verify schema has `type` field
- Check RJSF console for validation errors

### **Widget Wrong Type**

- Check schema `type` and `format` fields
- Verify widget is registered in `widgets/index.ts`
- Check if custom field is overriding default behavior

### **Handle Not Appearing**

- Check `showHandles` in formContext
- Verify not inside `fromAnyOf` context
- Check if field is credential or array item

### **Value Not Saving**

- Verify `onChange` callback is firing
- Check `handleChange` in FormCreator
- Look for console errors in `updateNodeData`

---

## Summary

The Input-Renderer is a sophisticated form system that:

1. **Uses RJSF** as the foundation for JSON Schema → React forms
2. **Extends RJSF** with custom Fields, Templates, and Widgets
3. **Integrates** with FlowEditor's connection system
4. **Handles** complex types (anyOf, credentials, free-form objects)
5. **Provides** connection-aware, type-safe input rendering

**Key Hierarchy (What Comes First):**

```
JSON Schema
  → Pre-processing
    → RJSF Form
      → SchemaField (RJSF internal)
        → Field (AnyOfField, CredentialsField, etc.)
          → Template (FieldTemplate, ArrayFieldTemplate)
            → Widget (TextWidget, SelectWidget, etc.)
              → HTML Element
```

**Mental Model:**

- **Fields** = Smart logic layers (type selection, OAuth flows)
- **Templates** = Layout wrappers (handles, labels, tooltips)
- **Widgets** = Actual inputs (text boxes, dropdowns)

**Integration Point:**

- FormRenderer receives schema from `node.data.inputSchema`
- User edits form → `onChange` → `nodeStore.updateNodeData()`
- Values saved as `node.data.hardcodedValues`
