import type { Meta, StoryObj } from "@storybook/nextjs";
import { Breadcrumbs } from "./Breadcrumbs";

const meta: Meta<typeof Breadcrumbs> = {
  title: "Molecules/Breadcrumbs",
  component: Breadcrumbs,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: `
## Breadcrumbs Component

A navigation breadcrumb component that shows the current page's location within a navigational hierarchy.

### ✨ Features

- **Clean design** - Simple, elegant breadcrumb navigation
- **Link integration** - Uses custom Link component for navigation
- **Responsive** - Flexible layout that wraps on smaller screens
- **Accessible** - Semantic markup with proper link structure
- **TypeScript support** - Full TypeScript interface support
- **Customizable** - Easy to style and extend

### 🎯 Usage

\`\`\`tsx
<Breadcrumbs 
  items={[
    { name: "Home", link: "/" },
    { name: "Library", link: "/library" },
    { name: "Agents", link: "/library/agents" },
  ]} 
/>
\`\`\`

### Props

- **items**: Array of breadcrumb items with name and link properties

### BreadcrumbItem Interface

\`\`\`tsx
interface BreadcrumbItem {
  name: string;  // Display text for the breadcrumb
  link: string;  // URL to navigate to when clicked
}
\`\`\`
        `,
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    items: {
      control: "object",
      description: "Array of breadcrumb items with name and link properties",
      table: {
        type: {
          summary: "BreadcrumbItem[]",
          detail: "Array of { name: string, link: string }",
        },
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Basic breadcrumb navigation with a few levels.
 */
export const Default: Story = {
  args: {
    items: [
      { name: "Home", link: "/" },
      { name: "Library", link: "/library" },
      { name: "Agents", link: "/library/agents" },
    ],
  },
};

/**
 * Single breadcrumb item (just home).
 */
export const Single: Story = {
  args: {
    items: [{ name: "Home", link: "/" }],
  },
};

/**
 * Two-level breadcrumb navigation.
 */
export const TwoLevels: Story = {
  args: {
    items: [
      { name: "Dashboard", link: "/dashboard" },
      { name: "Settings", link: "/dashboard/settings" },
    ],
  },
};

/**
 * Deep navigation with many levels.
 */
export const DeepNavigation: Story = {
  args: {
    items: [
      { name: "Home", link: "/" },
      { name: "Platform", link: "/platform" },
      { name: "Library", link: "/platform/library" },
      { name: "Agents", link: "/platform/library/agents" },
      { name: "My Agent", link: "/platform/library/agents/123" },
      { name: "Edit", link: "/platform/library/agents/123/edit" },
    ],
  },
};

/**
 * Breadcrumbs with longer text to show wrapping behavior.
 */
export const LongText: Story = {
  args: {
    items: [
      { name: "AutoGPT Platform", link: "/" },
      { name: "Agent Library Management", link: "/library" },
      { name: "Advanced Configuration Settings", link: "/library/settings" },
    ],
  },
};
