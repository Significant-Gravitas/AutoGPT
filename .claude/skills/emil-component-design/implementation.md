# Component Design — Implementation Patterns

Working code for the patterns described in SKILL.md.

## Compound components with context

Share implicit state through React context, not props — the root provides, the parts consume:

```jsx
const DialogContext = createContext(null);

function Dialog({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <DialogContext.Provider value={{ open, setOpen }}>
      {children}
    </DialogContext.Provider>
  );
}

function DialogTrigger(props) {
  const { setOpen } = useContext(DialogContext);
  return <button onClick={() => setOpen(true)} {...props} />;
}
```

## Controlled + uncontrolled in one component

Internal state by default, external state when the consumer provides `value`:

```jsx
function Input({
  value: controlledValue,
  defaultValue,
  onChange,
  ...props
}) {
  const [internalValue, setInternalValue] = useState(defaultValue ?? "");

  const isControlled = controlledValue !== undefined;
  const value = isControlled ? controlledValue : internalValue;

  function handleChange(e) {
    if (!isControlled) {
      setInternalValue(e.target.value);
    }
    onChange?.(e);
  }

  return <input value={value} onChange={handleChange} {...props} />;
}

// Uncontrolled
<Input defaultValue="hello" />

// Controlled
<Input value={value} onChange={setValue} />
```

## `asChild` with Radix Slot

```jsx
import { Slot } from "@radix-ui/react-slot";

function Button({ asChild, ...props }) {
  const Comp = asChild ? Slot : "button";
  return <Comp {...props} />;
}
```

## Forwarding refs

Always forward refs on components that wrap DOM elements:

```jsx
const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ children, ...props }, ref) => {
    return (
      <button ref={ref} {...props}>
        {children}
      </button>
    );
  }
);
```

## Spreading remaining props

Allow arbitrary HTML attributes to pass through:

```jsx
function Button({ variant, size, className, ...props }) {
  return (
    <button
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
}

// Now these work:
<Button data-testid="submit" aria-label="Submit form">
  Submit
</Button>
```

## Sensible defaults

Defaults should cover 80% of usages:

```jsx
function Button({
  variant = "primary",
  size = "md",
  type = "button", // Not "submit" - safer default
  ...props
}) {
  // ...
}
```

## Slot pattern for optional sections

```jsx
function Card({ children, header, footer }) {
  return (
    <div className="card">
      {header && <div className="card-header">{header}</div>}
      <div className="card-content">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

// Usage
<Card
  header={<h2>Title</h2>}
  footer={<Button>Save</Button>}
>
  Main content
</Card>
```

## Error boundaries

Wrap complex components in error boundaries so one broken widget doesn't take down the page:

```jsx
function ComponentWithErrorBoundary({ children }) {
  return (
    <ErrorBoundary fallback={<ErrorFallback />}>
      {children}
    </ErrorBoundary>
  );
}
```

## File structure and exports

```
components/
├── button/
│   ├── button.tsx        # Main component
│   ├── button.test.tsx   # Tests
│   └── index.ts          # Public exports
├── card/
│   ├── card.tsx
│   ├── card-header.tsx
│   ├── card-content.tsx
│   └── index.ts
```

```tsx
// components/card/index.ts
export { Card } from "./card";
export { CardHeader } from "./card-header";
export { CardContent } from "./card-content";

// Or as compound component
export { Card, CardHeader, CardContent } from "./card";
```
