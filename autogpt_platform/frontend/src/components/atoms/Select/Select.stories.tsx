import type { Meta, StoryObj } from "@storybook/nextjs";
import { Select } from "./Select";

const meta: Meta<typeof Select> = {
  title: "Atoms/Select",
  tags: ["autodocs"],
  component: Select,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Select component based on our design system. Built on shadcn/ui with styling that matches our Input. Supports size variants (small | medium) and optional hidden label.",
      },
    },
  },
  argTypes: {
    placeholder: {
      control: "text",
      description: "Placeholder text",
    },
    value: {
      control: "text",
      description: "The selected value",
    },
    label: {
      control: "text",
      description:
        "Label text (used as placeholder if no placeholder provided)",
    },
    disabled: {
      control: "boolean",
      description: "Disable the select",
    },
    hideLabel: {
      control: "boolean",
      description: "Hide the label",
    },
    error: {
      control: "text",
      description: "Error message to display below the select",
    },
    options: {
      control: "object",
      description: "Array of options with value and label properties",
    },
    size: {
      control: { type: "radio" },
      options: ["small", "medium"],
      description:
        "Visual size variant. small = compact trigger (22px line-height), medium = default (46px height).",
    },
  },
  args: {
    placeholder: "Select an option...",
    value: "",
    disabled: false,
    hideLabel: false,
    options: [
      { value: "option1", label: "Option 1" },
      { value: "option2", label: "Option 2" },
      { value: "option3", label: "Option 3" },
    ],
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Basic variants
export const Default: Story = {
  args: {
    label: "Country",
    placeholder: "Select a country",
    options: [
      { value: "us", label: "United States" },
      { value: "ca", label: "Canada" },
      { value: "uk", label: "United Kingdom" },
      { value: "de", label: "Germany" },
      { value: "fr", label: "France" },
    ],
  },
};

export const WithValue: Story = {
  args: {
    label: "Country",
    placeholder: "Select a country",
    value: "us",
    options: [
      { value: "us", label: "United States" },
      { value: "ca", label: "Canada" },
      { value: "uk", label: "United Kingdom" },
      { value: "de", label: "Germany" },
      { value: "fr", label: "France" },
    ],
  },
};

export const Small: Story = {
  args: {
    id: "select-small",
    label: "Compact",
    hideLabel: true,
    size: "small",
    placeholder: "Choose option",
    options: [
      { value: "opt1", label: "Option 1" },
      { value: "opt2", label: "Option 2" },
      { value: "opt3", label: "Option 3" },
    ],
  },
  parameters: {
    docs: {
      description: {
        story:
          "Small size is ideal for dense UIs (e.g., inline controls like TimePicker).",
      },
    },
  },
};

export const Medium: Story = {
  args: {
    id: "select-medium",
    label: "Medium",
    size: "medium",
    placeholder: "Choose option",
    options: [
      { value: "opt1", label: "Option 1" },
      { value: "opt2", label: "Option 2" },
      { value: "opt3", label: "Option 3" },
    ],
  },
};

export const WithIconsAndSeparators: Story = {
  render: function IconsStory() {
    const opts = [
      { value: "oauth", label: "Your Google account", icon: <span>✅</span> },
      { separator: true, value: "sep1", label: "" } as any,
      {
        value: "signin",
        label: "Sign in with Google",
        icon: <span>🔐</span>,
        onSelect: () => alert("Sign in"),
      },
      {
        value: "add-key",
        label: "Add API key",
        onSelect: () => alert("Add key"),
      },
    ];
    return (
      <div className="w-[320px]">
        <Select
          id="rich"
          label="Rich"
          hideLabel
          options={opts as any}
          placeholder="Choose"
        />
      </div>
    );
  },
  parameters: {
    docs: {
      description: {
        story:
          "Demonstrates icons, separators, and actionable rows via onSelect. onSelect prevents value change and triggers the action.",
      },
    },
  },
};

export const WithRenderItem: Story = {
  render: function RenderItemStory() {
    const opts = [
      { value: "opt1", label: "Option 1" },
      { value: "opt2", label: "Option 2", disabled: true },
      { value: "opt3", label: "Option 3" },
    ];
    return (
      <div className="w-[320px]">
        <Select
          id="render"
          label="Custom"
          hideLabel
          options={opts}
          placeholder="Pick one"
          renderItem={(o) => (
            <div className="flex items-center gap-2">
              <span className="font-medium">{o.label}</span>
              {o.disabled && (
                <span className="text-xs text-zinc-400">(disabled)</span>
              )}
            </div>
          )}
        />
      </div>
    );
  },
  parameters: {
    docs: {
      description: {
        story:
          "Custom rendering for options via renderItem prop; disabled items are styled and non-selectable.",
      },
    },
  },
};

export const WithoutLabel: Story = {
  args: {
    label: "Country",
    hideLabel: true,
    placeholder: "Select a country",
    options: [
      { value: "us", label: "United States" },
      { value: "ca", label: "Canada" },
      { value: "uk", label: "United Kingdom" },
      { value: "de", label: "Germany" },
      { value: "fr", label: "France" },
    ],
  },
};

export const Disabled: Story = {
  args: {
    label: "Country",
    placeholder: "Select a country",
    disabled: true,
    options: [
      { value: "us", label: "United States" },
      { value: "ca", label: "Canada" },
      { value: "uk", label: "United Kingdom" },
      { value: "de", label: "Germany" },
      { value: "fr", label: "France" },
    ],
  },
};

export const WithError: Story = {
  args: {
    label: "Country",
    placeholder: "Select a country",
    error: "Please select a valid country",
    options: [
      { value: "us", label: "United States" },
      { value: "ca", label: "Canada" },
      { value: "uk", label: "United Kingdom" },
      { value: "de", label: "Germany" },
      { value: "fr", label: "France" },
    ],
  },
};

export const ManyOptions: Story = {
  args: {
    label: "Programming Language",
    placeholder: "Select a programming language",
    options: [
      { value: "javascript", label: "JavaScript" },
      { value: "typescript", label: "TypeScript" },
      { value: "python", label: "Python" },
      { value: "java", label: "Java" },
      { value: "csharp", label: "C#" },
      { value: "cpp", label: "C++" },
      { value: "rust", label: "Rust" },
      { value: "go", label: "Go" },
      { value: "php", label: "PHP" },
      { value: "ruby", label: "Ruby" },
      { value: "swift", label: "Swift" },
      { value: "kotlin", label: "Kotlin" },
    ],
  },
};

export const AllVariants: Story = {
  render: renderAllVariants,
  parameters: {
    controls: {
      disable: true,
    },
    docs: {
      description: {
        story: "Complete showcase of all Select component variants and states.",
      },
    },
  },
};

function renderAllVariants() {
  const options = [
    { value: "us", label: "United States" },
    { value: "ca", label: "Canada" },
    { value: "uk", label: "United Kingdom" },
    { value: "de", label: "Germany" },
    { value: "fr", label: "France" },
  ];

  return (
    <div className="w-full max-w-md space-y-8">
      <Select
        label="Default Select"
        placeholder="Select a country"
        id="default"
        options={options}
      />

      <Select
        label="With Selected Value"
        placeholder="Select a country"
        value="us"
        id="with-value"
        options={options}
      />

      <Select
        label="Hidden Label"
        hideLabel={true}
        placeholder="Select a country"
        id="hidden-label"
        options={options}
      />

      <Select
        label="Disabled Select"
        placeholder="Select a country"
        disabled={true}
        id="disabled"
        options={options}
      />

      <Select
        label="With Error"
        placeholder="Select a country"
        error="Please select a valid country"
        id="with-error"
        options={options}
      />
    </div>
  );
}
