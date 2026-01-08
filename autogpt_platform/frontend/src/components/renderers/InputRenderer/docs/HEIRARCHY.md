# Input Renderer 2 - Hierarchy

## Flow Overview

```
FormRenderer2 → Form (RJSF) → ObjectFieldTemplate → FieldTemplate → Widget/Field
```

---

## Component Layers

### 1. Root (FormRenderer2)

- Entry point
- Preprocesses schema
- Passes to RJSF Form

### 2. Form (registry/Form.tsx)

- RJSF themed form
- Combines: templates + widgets + fields

### 3. Templates (decide layout/structure)

| Template                   | When Used                                   |
| -------------------------- | ------------------------------------------- |
| `ObjectFieldTemplate`      | `type: "object"`                            |
| `ArrayFieldTemplate`       | `type: "array"`                             |
| `FieldTemplate`            | Wraps every field (title, errors, children) |
| `ArrayFieldItemTemplate`   | Each array item                             |
| `WrapIfAdditionalTemplate` | Additional properties in objects            |

### 4. Fields (custom rendering logic)

| Field              | When Used                    |
| ------------------ | ---------------------------- |
| `AnyOfField`       | `anyOf` or `oneOf` in schema |
| `ArraySchemaField` | Array type handling          |

### 5. Widgets (actual input elements)

| Widget           | Input Type              |
| ---------------- | ----------------------- |
| `TextWidget`     | string, number, integer |
| `SelectWidget`   | enum, anyOf selector    |
| `CheckboxWidget` | boolean                 |
| `FileWidget`     | file upload             |
| `DateWidget`     | date                    |
| `TimeWidget`     | time                    |
| `DateTimeWidget` | datetime                |

---

## Your Schema Hierarchy

```
Root (type: object)
└── ObjectFieldTemplate
    │
    ├── name (string, required)
    │   └── FieldTemplate → TextWidget
    │
    ├── value (anyOf)
    │   └── FieldTemplate → AnyOfField
    │       └── Selector dropdown + selected type:
    │           ├── String → TextWidget
    │           ├── Number → TextWidget
    │           ├── Integer → TextWidget
    │           ├── Boolean → CheckboxWidget
    │           ├── Array → ArrayFieldTemplate → items
    │           ├── Object → ObjectFieldTemplate
    │           └── Null → nothing
    │
    ├── title (anyOf: string | null)
    │   └── FieldTemplate → AnyOfField
    │       └── String → TextWidget OR Null → nothing
    │
    ├── description (anyOf: string | null)
    │   └── FieldTemplate → AnyOfField
    │       └── String → TextWidget OR Null → nothing
    │
    ├── placeholder_values (array of strings)
    │   └── FieldTemplate → ArrayFieldTemplate
    │       └── ArrayFieldItemTemplate (per item)
    │           └── TextWidget
    │
    ├── advanced (boolean)
    │   └── FieldTemplate → CheckboxWidget
    │
    └── secret (boolean)
        └── FieldTemplate → CheckboxWidget
```

---

## Nested Examples (up to 3 levels)

### Simple Array (strings)

```json
{ "tags": { "type": "array", "items": { "type": "string" } } }
```

```
Level 1: ObjectFieldTemplate (root)
└── Level 2: FieldTemplate → ArrayFieldTemplate
    └── Level 3: ArrayFieldItemTemplate → TextWidget
```

### Array of Objects

```json
{
  "users": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" }
      }
    }
  }
}
```

```
Level 1: ObjectFieldTemplate (root)
└── Level 2: FieldTemplate → ArrayFieldTemplate
    └── Level 3: ArrayFieldItemTemplate → ObjectFieldTemplate
        ├── FieldTemplate → TextWidget (name)
        └── FieldTemplate → TextWidget (age)
```

### Nested Object (3 levels)

```json
{
  "config": {
    "type": "object",
    "properties": {
      "database": {
        "type": "object",
        "properties": {
          "host": { "type": "string" },
          "port": { "type": "integer" }
        }
      }
    }
  }
}
```

```
Level 1: ObjectFieldTemplate (root)
└── config
    └── Level 2: FieldTemplate → ObjectFieldTemplate
        └── database
            └── Level 3: FieldTemplate → ObjectFieldTemplate
                ├── FieldTemplate → TextWidget (host)
                └── FieldTemplate → TextWidget (port)
```

### Array of Arrays (nested array)

```json
{
  "matrix": {
    "type": "array",
    "items": {
      "type": "array",
      "items": { "type": "number" }
    }
  }
}
```

```
Level 1: ObjectFieldTemplate (root)
└── Level 2: FieldTemplate → ArrayFieldTemplate
    └── Level 3: ArrayFieldItemTemplate → ArrayFieldTemplate
        └── ArrayFieldItemTemplate → TextWidget
```

### Complex: Object → Array → Object

```json
{
  "company": {
    "type": "object",
    "properties": {
      "departments": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "budget": { "type": "number" }
          }
        }
      }
    }
  }
}
```

```
Level 1: ObjectFieldTemplate (root)
└── company
    └── Level 2: FieldTemplate → ObjectFieldTemplate
        └── departments
            └── Level 3: FieldTemplate → ArrayFieldTemplate
                └── ArrayFieldItemTemplate → ObjectFieldTemplate
                    ├── FieldTemplate → TextWidget (name)
                    └── FieldTemplate → TextWidget (budget)
```

### anyOf inside Array

```json
{
  "items": {
    "type": "array",
    "items": {
      "anyOf": [
        { "type": "string" },
        { "type": "object", "properties": { "id": { "type": "string" } } }
      ]
    }
  }
}
```

```
Level 1: ObjectFieldTemplate (root)
└── Level 2: FieldTemplate → ArrayFieldTemplate
    └── Level 3: ArrayFieldItemTemplate → AnyOfField
        └── Selector + selected:
            ├── String → TextWidget
            └── Object → ObjectFieldTemplate
                └── FieldTemplate → TextWidget (id)
```

---

## Nesting Pattern Summary

| Parent Type | Child Wrapper                                   |
| ----------- | ----------------------------------------------- |
| object      | `ObjectFieldTemplate` → `FieldTemplate`         |
| array       | `ArrayFieldTemplate` → `ArrayFieldItemTemplate` |
| anyOf       | `AnyOfField` → selected schema's template       |
| primitive   | `Widget` (leaf - no children)                   |

**Pattern:** Each level adds FieldTemplate wrapper except array items (use ArrayFieldItemTemplate)

---

## Key Points

1. **FieldTemplate wraps everything** - handles title, description, errors
2. **anyOf = AnyOfField** - shows dropdown to pick type, then renders selected schema
3. **ObjectFieldTemplate loops properties** - each property gets FieldTemplate
4. **ArrayFieldTemplate loops items** - each item gets ArrayFieldItemTemplate
5. **Widgets are leaf nodes** - actual input controls user interacts with
6. **Nesting repeats the pattern** - object/array/anyOf can contain object/array/anyOf recursively

---

## Decision Flow

```
Schema Type?
├── object → ObjectFieldTemplate → loop properties
├── array → ArrayFieldTemplate → loop items
├── anyOf/oneOf → AnyOfField → selector + selected schema
└── primitive (string/number/boolean) → Widget
```

---

## Template Wrapping Order

```
ObjectFieldTemplate (root)
└── FieldTemplate (per property)
    └── WrapIfAdditionalTemplate (if additionalProperties)
        └── TitleField + DescriptionField + children
            └── Widget OR nested Template/Field
```
