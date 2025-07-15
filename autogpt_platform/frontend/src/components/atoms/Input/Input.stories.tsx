import type { Meta, StoryObj } from "@storybook/nextjs";
import { Input } from "./Input";

const meta: Meta<typeof Input> = {
  title: "Atoms/Input",
  tags: ["autodocs"],
  component: Input,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Input component based on our design system. Built on top of shadcn/ui input with custom styling matching Figma designs.",
      },
    },
  },
  argTypes: {
    type: {
      control: "select",
      options: ["text", "email", "password", "number", "amount", "tel", "url"],
      description: "Input type",
    },
    placeholder: {
      control: "text",
      description: "Placeholder text",
    },
    value: {
      control: "text",
      description: "The value of the input",
    },
    label: {
      control: "text",
      description:
        "Label text (used as placeholder if no placeholder provided)",
    },
    disabled: {
      control: "boolean",
      description: "Disable the input",
    },
    hideLabel: {
      control: "boolean",
      description: "Hide the label",
    },
    decimalCount: {
      control: "number",
      description:
        "Number of decimal places allowed (only for amount type). Default is 4.",
    },
    error: {
      control: "text",
      description: "Error message to display below the input",
    },
  },
  args: {
    placeholder: "Enter text...",
    type: "text",
    value: "",
    disabled: false,
    hideLabel: false,
    decimalCount: 4,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Basic variants
export const Default: Story = {
  args: {
    placeholder: "Enter your text",
    label: "Full Name",
  },
};

export const WithoutLabel: Story = {
  args: {
    label: "Full Name",
    hideLabel: true,
  },
};

export const Disabled: Story = {
  args: {
    placeholder: "This field is disabled",
    label: "Full Name",
    disabled: true,
  },
};

export const WithError: Story = {
  args: {
    label: "Email",
    type: "email",
    placeholder: "Enter your email",
    error: "Please enter a valid email address",
  },
};

export const InputTypes: Story = {
  render: renderInputTypes,
  parameters: {
    controls: {
      disable: true,
    },
    docs: {
      description: {
        story:
          "Complete showcase of all input types with their specific behaviors. Test each input type to verify filtering and formatting works correctly.",
      },
    },
  },
};

// Render functions as function declarations
function renderInputTypes() {
  return (
    <div className="w-full max-w-md space-y-8">
      <Input
        label="Full Name"
        type="text"
        placeholder="Enter your full name"
        id="full-name"
      />
      <Input
        label="Email"
        type="email"
        placeholder="your.email@example.com"
        id="email"
      />
      <Input
        label="Password"
        type="password"
        placeholder="Enter your password"
        id="password"
      />
      <div className="flex flex-col gap-4">
        <p className="font-mono text-sm">
          If type=&quot;number&quot; prop is provided, the input will allow only
          positive or negative numbers. No decimal limiting.
        </p>
        <Input
          label="Age"
          type="number"
          placeholder="Enter your age"
          id="age"
        />
      </div>
      <div className="flex flex-col gap-4">
        <p className="font-mono text-sm">
          If type=&quot;amount&quot; prop is provided, it formats numbers with
          commas (1000 → 1,000) and limits decimals via decimalCount prop.
        </p>
        <Input
          label="Price"
          type="amount"
          placeholder="Enter amount"
          decimalCount={2}
          id="price"
        />
      </div>
      <div className="flex flex-col gap-4">
        <p className="font-mono text-sm">
          If type=&quot;tel&quot; prop is provided, the input will allow only
          numbers, spaces, parentheses (), plus +, and brackets [].
        </p>
        <Input
          label="Phone"
          type="tel"
          placeholder="+1 (555) 123-4567"
          id="phone"
        />
      </div>
      <Input
        label="Website"
        type="url"
        placeholder="https://example.com"
        id="website"
      />
    </div>
  );
}
