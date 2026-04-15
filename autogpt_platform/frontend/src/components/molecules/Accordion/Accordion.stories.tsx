import type { Meta } from "@storybook/nextjs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./Accordion";

const meta: Meta<typeof Accordion> = {
  title: "Molecules/Accordion",
  component: Accordion,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: `
## Accordion Component

A vertically stacked set of interactive headings that each reveal an associated section of content.

### ✨ Features

- **Built on Radix UI** - Uses @radix-ui/react-accordion for accessibility and functionality
- **Single or multiple** - Supports single or multiple items open at once
- **Smooth animations** - Built-in expand/collapse animations
- **Accessible** - Full keyboard navigation and screen reader support
- **Customizable** - Style with Tailwind CSS classes

### 🎯 Usage

\`\`\`tsx
<Accordion type="single" collapsible>
  <AccordionItem value="item-1">
    <AccordionTrigger>Is it accessible?</AccordionTrigger>
    <AccordionContent>
      Yes. It adheres to the WAI-ARIA design pattern.
    </AccordionContent>
  </AccordionItem>
</Accordion>
\`\`\`

### Props

**Accordion**:
- **type**: "single" | "multiple" - Whether one or multiple items can be open
- **collapsible**: boolean - When type is "single", allows closing all items
- **defaultValue**: string | string[] - Default open item(s)
- **value**: string | string[] - Controlled open item(s)
- **onValueChange**: (value) => void - Callback when value changes

**AccordionItem**:
- **value**: string - Unique identifier for the item
- **disabled**: boolean - Whether the item is disabled

**AccordionTrigger**:
- Standard button props

**AccordionContent**:
- Standard div props
        `,
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    type: {
      control: "radio",
      options: ["single", "multiple"],
      description: "Whether one or multiple items can be open at the same time",
      table: {
        defaultValue: { summary: "single" },
      },
    },
    collapsible: {
      control: "boolean",
      description:
        'When type is "single", allows closing content when clicking on open trigger',
      table: {
        defaultValue: { summary: "false" },
      },
    },
  },
};

export default meta;

export function Default() {
  return (
    <Accordion type="single" collapsible className="w-96">
      <AccordionItem value="item-1">
        <AccordionTrigger>Is it accessible?</AccordionTrigger>
        <AccordionContent>
          Yes. It adheres to the WAI-ARIA design pattern.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2">
        <AccordionTrigger>Is it styled?</AccordionTrigger>
        <AccordionContent>
          Yes. It comes with default styles that match your design system.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-3">
        <AccordionTrigger>Is it animated?</AccordionTrigger>
        <AccordionContent>
          Yes. It&apos;s animated by default with smooth expand/collapse
          transitions.
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

export function Multiple() {
  return (
    <Accordion type="multiple" className="w-96">
      <AccordionItem value="item-1">
        <AccordionTrigger>First section</AccordionTrigger>
        <AccordionContent>
          Multiple items can be open at the same time when type is set to
          &quot;multiple&quot;.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2">
        <AccordionTrigger>Second section</AccordionTrigger>
        <AccordionContent>
          Try opening this one while the first is still open.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-3">
        <AccordionTrigger>Third section</AccordionTrigger>
        <AccordionContent>
          All three can be open simultaneously.
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

export function DefaultOpen() {
  return (
    <Accordion type="single" collapsible defaultValue="item-2" className="w-96">
      <AccordionItem value="item-1">
        <AccordionTrigger>Closed by default</AccordionTrigger>
        <AccordionContent>This item starts closed.</AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2">
        <AccordionTrigger>Open by default</AccordionTrigger>
        <AccordionContent>
          This item starts open because defaultValue is set to
          &quot;item-2&quot;.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-3">
        <AccordionTrigger>Also closed</AccordionTrigger>
        <AccordionContent>This item also starts closed.</AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

export function WithDisabledItem() {
  return (
    <Accordion type="single" collapsible className="w-96">
      <AccordionItem value="item-1">
        <AccordionTrigger>Available item</AccordionTrigger>
        <AccordionContent>This item can be toggled.</AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2" disabled>
        <AccordionTrigger>Disabled item</AccordionTrigger>
        <AccordionContent>
          This content cannot be accessed because the item is disabled.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-3">
        <AccordionTrigger>Another available item</AccordionTrigger>
        <AccordionContent>This item can also be toggled.</AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

export function CustomStyled() {
  return (
    <Accordion type="single" collapsible className="w-96">
      <AccordionItem value="item-1" className="border-none">
        <AccordionTrigger className="rounded-lg bg-zinc-100 px-4 hover:bg-zinc-200 hover:no-underline">
          Custom styled trigger
        </AccordionTrigger>
        <AccordionContent className="px-4 pt-2">
          You can customize the styling using className props.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2" className="mt-2 border-none">
        <AccordionTrigger className="rounded-lg bg-blue-50 px-4 text-blue-700 hover:bg-blue-100 hover:no-underline">
          Blue themed
        </AccordionTrigger>
        <AccordionContent className="px-4 pt-2 text-blue-600">
          Each item can have different styles.
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
