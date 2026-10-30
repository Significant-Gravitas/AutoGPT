---
name: emil-component-design
description: "Design React component APIs that are composable, predictable, and hard to misuse — composition over configuration, compound components, controlled/uncontrolled state, and sensible props. Use when creating or refactoring React components, designing a component library, reviewing a props API, or deciding how a component should expose customization. Triggers on: component API, props design, compound components, composition, configuration props, controlled, uncontrolled, defaultValue, asChild, Slot, render props, forwardRef, ref forwarding, variants, className override, design system components, reusable components, component library, Radix, shadcn, prop drilling, boolean props, slots, children, context."
---

# Component Design

A component API is a contract. Good ones make the common case a one-liner and the uncommon case possible without forking. The failure modes are always the same: too rigid (users fork or hack around you) or too configurable (30 props nobody can hold in their head). Reach for composition before configuration, name things the way the platform names them, and support both controlled and uncontrolled state when state is involved.

## Quick Reference

| Topic | Where |
|---|---|
| Core API decisions (composition, compound, props, state) | This file |
| Implementation patterns (context, controlled/uncontrolled, asChild, refs, slots, file structure) | [implementation.md](implementation.md) |

## Core Principles

### 1. Composition over configuration

Expose structure as JSX children, not configuration objects:

```jsx
// Good - composable
<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
    <CardDescription>Description</CardDescription>
  </CardHeader>
  <CardContent>Content here</CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>

// Bad - configuration object
<Card
  header={{ title: "Title", description: "Description" }}
  content="Content here"
  footer={{ actions: [{ label: "Action", onClick: () => {} }] }}
/>
```

### 2. Compound components for multi-part UI

When a component has multiple related parts sharing implicit state, split it into compound components instead of prop-drilling everything through one root:

```jsx
// Good - compound components
<Dialog>
  <Dialog.Trigger>Open</Dialog.Trigger>
  <Dialog.Content>
    <Dialog.Title>Are you sure?</Dialog.Title>
    <Dialog.Description>This action cannot be undone.</Dialog.Description>
    <Dialog.Close>Cancel</Dialog.Close>
  </Dialog.Content>
</Dialog>

// Bad - prop drilling everything
<Dialog
  trigger="Open"
  title="Are you sure?"
  description="This action cannot be undone."
  closeText="Cancel"
/>
```

Use compound components when:
- Multiple related elements share implicit state
- The component has slots (header, body, footer)
- The order or presence of children varies between usages
- Consumers need flexible composition

Don't use them when:
- The component is simple with a fixed structure
- It has 1-3 props
- The structure never changes

Share the implicit state through React context — the root provides, the parts consume. Implementation in [implementation.md](implementation.md).

### 3. The Goldilocks level of customization

Too little customization and users fork your component or add hacky workarounds. Too much and the API becomes confusing and a maintenance nightmare. Aim for the middle:

```jsx
// Too rigid - no customization
<Button>Click me</Button>

// Too flexible - overwhelming API
<Button
  backgroundColor="#000"
  hoverBackgroundColor="#333"
  activeBackgroundColor="#111"
  borderRadius={4}
  paddingX={16}
  paddingY={8}
  fontSize={14}
  fontWeight={500}
  // ... 30 more props
>
  Click me
</Button>

// Just right - variants + escape hatch
<Button variant="primary" size="md" className="custom-override">
  Click me
</Button>
```

Layer customization in this order:
1. **Variants** — predefined options (`primary`, `secondary`, `destructive`)
2. **Size** — predefined sizes (`sm`, `md`, `lg`)
3. **className** — escape hatch for one-off overrides
4. **asChild** — render as a different element (Radix pattern)

When implementing variants, express the styles in the project's existing styling system (Tailwind, CSS Modules, vanilla-extract, styled-components — whatever the codebase already uses). Don't impose a new one.

### 4. Name props like the platform does

Be consistent across every component, mirror HTML where possible:

```jsx
// Good - consistent patterns
<Input disabled />
<Button disabled />
<Select disabled />

// Bad - inconsistent
<Input disabled />
<Button isDisabled />
<Select readonly />
```

Boolean props get positive names — never double negatives:

```jsx
// Good
<Input disabled />
<Modal open />

// Bad
<Input notEnabled />
<Modal isNotClosed />
```

Event handlers are prefixed with `on`:

```jsx
// Good
<Input onChange={} onBlur={} />
<Dialog onOpenChange={} onClose={} />

// Bad
<Input handleChange={} blurHandler={} />
```

### 5. Support both controlled and uncontrolled state

Any stateful component should work both ways — internal state by default, external state when the consumer needs it:

```jsx
// Uncontrolled
<Input defaultValue="hello" />

// Controlled
<Input value={value} onChange={setValue} />
```

Full dual-mode implementation in [implementation.md](implementation.md).

### 6. Play nice with the DOM

- **Forward refs** on every component that wraps a DOM element — otherwise focus management, tooltips, and popover positioning break for composers.
- **Spread remaining props** (`{...props}`) so `aria-*`, `data-testid`, and other HTML attributes pass through.
- **Pick defaults that work for 80% of usages** — e.g. `variant = "primary"`, `size = "md"`, and `type = "button"` (not `"submit"` — the safer default).

Code for all three in [implementation.md](implementation.md).

### 7. The `asChild` pattern

Let consumers swap the rendered element while keeping the behavior and styles:

```jsx
// Render as button (default)
<Button>Click me</Button>

// Render as link
<Button asChild>
  <a href="/page">Click me</a>
</Button>

// Render as Next.js Link
<Button asChild>
  <Link href="/page">Click me</Link>
</Button>
```

### 8. Children for simple content, render props for data

```jsx
// Simple - use children
<Card>
  <CardContent />
</Card>

// Complex with data - render prop
<List
  items={users}
  renderItem={(user) => <UserCard user={user} />}
/>
```

For optional sections that take arbitrary JSX, use slot props (`header={<h2>Title</h2>}`) — see the slot pattern in [implementation.md](implementation.md).

## Common Mistakes

- **Prop explosion.** `leftIcon`, `rightIcon`, `iconSpacing`, `iconSize`... just use children: `<Button><Icon /> Click <Arrow /></Button>`.
- **Boolean soup instead of variants.** `<Button primary large rounded>` invites invalid combinations. Use `<Button variant="primary" size="lg" radius="full">`.
- **Premature abstraction.** Don't create a shared component until you've copy-pasted the code 2-3 times. Wait for the pattern to emerge, then extract it.
- **Swallowing props.** Destructuring everything and forgetting `{...props}` breaks `aria-*`, `data-*`, and testing attributes.
- **Missing ref forwarding.** Breaks focus management, tooltips, and popover positioning for anyone composing your component.

## Checklist

- [ ] Structure expressed as JSX children, not config objects
- [ ] Multi-part components use compound parts with context, not prop drilling
- [ ] Variants + size + `className` escape hatch; no style-prop explosion
- [ ] Prop names match HTML conventions and are consistent across components
- [ ] Stateful components support both `value`/`onChange` and `defaultValue`
- [ ] Refs forwarded, remaining props spread, `type="button"` default
- [ ] Styling implemented in the project's existing styling system
