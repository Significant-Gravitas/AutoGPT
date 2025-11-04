import type { Meta, StoryObj } from "@storybook/nextjs";
import { MarkdownContent } from "./MarkdownContent";

const meta = {
  title: "Atoms/MarkdownContent",
  component: MarkdownContent,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof MarkdownContent>;

export default meta;
type Story = StoryObj<typeof meta>;

export const BasicText: Story = {
  args: {
    content: "This is a simple paragraph with **bold text** and *italic text*.",
  },
};

export const InlineCode: Story = {
  args: {
    content:
      "Use the `useState` hook to manage state in React components. You can also use `useEffect` for side effects.",
  },
};

export const CodeBlock: Story = {
  args: {
    content: `Here's a code example:

\`\`\`typescript
function greet(name: string): string {
  return \`Hello, \${name}!\`;
}

const message = greet("World");
console.log(message);
\`\`\`

This is a TypeScript function that returns a greeting.`,
  },
};

export const Links: Story = {
  args: {
    content: `Check out these resources:
- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com)

All links open in new tabs for your convenience.`,
  },
};

export const UnorderedList: Story = {
  args: {
    content: `Shopping list:
- Apples
- Bananas
- Oranges
- Grapes
- Strawberries`,
  },
};

export const OrderedList: Story = {
  args: {
    content: `Steps to deploy:
1. Run tests locally
2. Create a pull request
3. Wait for CI to pass
4. Get code review approval
5. Merge to main
6. Deploy to production`,
  },
};

export const TaskList: Story = {
  args: {
    content: `Project tasks:
- [x] Set up project structure
- [x] Implement authentication
- [ ] Add user dashboard
- [ ] Create admin panel
- [ ] Write documentation`,
  },
};

export const Blockquote: Story = {
  args: {
    content: `As Einstein said:

> Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.

This quote reminds us to think creatively.`,
  },
};

export const Table: Story = {
  args: {
    content: `Here's a comparison table:

| Feature | Basic | Pro | Enterprise |
|---------|-------|-----|------------|
| Users | 5 | 50 | Unlimited |
| Storage | 10GB | 100GB | 1TB |
| Support | Email | Priority | 24/7 Phone |
| Price | $9/mo | $29/mo | Custom |`,
  },
};

export const Headings: Story = {
  args: {
    content: `# Heading 1
This is the largest heading.

## Heading 2
A bit smaller.

### Heading 3
Even smaller.

#### Heading 4
Getting smaller still.

##### Heading 5
Almost the smallest.

###### Heading 6
The smallest heading.`,
  },
};

export const StrikethroughAndFormatting: Story = {
  args: {
    content: `Text formatting options:
- **Bold text** is important
- *Italic text* is emphasized
- ~~Strikethrough~~ text is deleted
- ***Bold and italic*** is very important
- **Bold with *nested italic***`,
  },
};

export const HorizontalRule: Story = {
  args: {
    content: `Section One

---

Section Two

---

Section Three`,
  },
};

export const MixedContent: Story = {
  args: {
    content: `# Chat Message Example

I found **three solutions** to your problem:

## 1. Using the API

You can call the endpoint like this:

\`\`\`typescript
const response = await fetch('/api/users', {
  method: 'GET',
  headers: { 'Authorization': \`Bearer \${token}\` }
});
\`\`\`

## 2. Using the CLI

Alternatively, use the command line:

\`\`\`bash
cli users list --format json
\`\`\`

## 3. Manual approach

If you prefer, you can:
1. Open the dashboard
2. Navigate to *Users* section
3. Click **Export**
4. Choose JSON format

> **Note**: The API approach is recommended for automation.

For more information, check out the [documentation](https://docs.example.com).`,
  },
};

export const XSSAttempt: Story = {
  args: {
    content: `# Security Test

This content attempts XSS attacks that should be escaped:

<script>alert('XSS')</script>

<img src="x" onerror="alert('XSS')">

<a href="javascript:alert('XSS')">Click me</a>

<style>body { background: red; }</style>

All of these should render as plain text, not execute.`,
  },
};

export const MalformedMarkdown: Story = {
  args: {
    content: `# Unclosed Heading

**Bold without closing

\`\`\`
Code block without closing language tag

[Link with no URL]

![Image with no src]

**Nested *formatting without** proper closing*

| Table | with |
| mismatched | columns | extra |`,
  },
};

export const UnicodeAndEmoji: Story = {
  args: {
    content: `# Unicode Support

## Emojis
🎉 🚀 💡 ✨ 🔥 👍 ❤️ 🎯 📊 🌟

## Special Characters
→ ← ↑ ↓ © ® ™ € £ ¥ § ¶ † ‡

## Other Languages
你好世界 (Chinese)
مرحبا بالعالم (Arabic)
Привет мир (Russian)
हैलो वर्ल्ड (Hindi)

All characters should render correctly.`,
  },
};

export const LongCodeBlock: Story = {
  args: {
    content: `Here's a longer code example that tests overflow:

\`\`\`typescript
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;
  roles: string[];
  metadata: Record<string, unknown>;
}

function processUsers(users: User[]): Map<string, User> {
  return users.reduce((acc, user) => {
    acc.set(user.id, user);
    return acc;
  }, new Map<string, User>());
}

const users: User[] = [
  { id: '1', name: 'Alice', email: 'alice@example.com', createdAt: new Date(), updatedAt: new Date(), roles: ['admin'], metadata: {} },
  { id: '2', name: 'Bob', email: 'bob@example.com', createdAt: new Date(), updatedAt: new Date(), roles: ['user'], metadata: {} },
];

const userMap = processUsers(users);
console.log(userMap);
\`\`\`

The code block should scroll horizontally if needed.`,
  },
};

export const NestedStructures: Story = {
  args: {
    content: `# Nested Structures

## Lists within Blockquotes

> Here's a quote with a list:
> - First item
> - Second item
> - Third item

## Blockquotes within Lists

- Regular list item
- List item with quote:
  > This is a nested quote
- Another regular item

## Code in Lists

1. First step: Install dependencies
   \`\`\`bash
   npm install
   \`\`\`
2. Second step: Run the server
   \`\`\`bash
   npm start
   \`\`\`
3. Third step: Open browser`,
  },
};
