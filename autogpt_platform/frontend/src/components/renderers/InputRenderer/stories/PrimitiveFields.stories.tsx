import type { Meta, StoryObj } from "@storybook/nextjs";
import { FormRendererStory, storyDecorator } from "./FormRendererStoryWrapper";

const meta: Meta = {
  title: "Renderers/FormRenderer/Primitive Fields",
  tags: ["autodocs"],
  decorators: [storyDecorator],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Primitive field types: strings, numbers, booleans, enums, and date/time formats.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// --- String ---

export const StringField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          name: { type: "string", title: "Name", description: "Enter a name" },
        },
      }}
    />
  ),
};

export const StringWithDefault: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          greeting: {
            type: "string",
            title: "Greeting",
            default: "Hello, world!",
          },
        },
      }}
      initialValues={{ greeting: "Hello, world!" }}
    />
  ),
};

// --- Number ---

export const IntegerField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          count: { type: "integer", title: "Count", description: "A number" },
        },
      }}
    />
  ),
};

export const NumberField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          temperature: {
            type: "number",
            title: "Temperature",
            description: "A float value",
          },
        },
      }}
    />
  ),
};

export const NumberWithConstraints: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          score: {
            type: "number",
            title: "Score",
            minimum: 0,
            maximum: 100,
            description: "Value between 0 and 100",
          },
        },
      }}
    />
  ),
};

// --- Boolean ---

export const BooleanField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          enabled: {
            type: "boolean",
            title: "Enabled",
            description: "Toggle this on or off",
          },
        },
      }}
    />
  ),
};

// --- Enum ---

export const EnumField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          color: {
            type: "string",
            title: "Color",
            enum: ["red", "green", "blue", "yellow"],
          },
        },
      }}
    />
  ),
};

export const EnumWithDefault: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          priority: {
            type: "string",
            title: "Priority",
            enum: ["low", "medium", "high", "critical"],
            default: "medium",
          },
        },
      }}
      initialValues={{ priority: "medium" }}
    />
  ),
};

// --- Date / Time ---

export const DateField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          start_date: {
            type: "string",
            title: "Start Date",
            format: "date",
          },
        },
      }}
    />
  ),
};

export const TimeField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          alarm_time: {
            type: "string",
            title: "Alarm Time",
            format: "time",
          },
        },
      }}
    />
  ),
};

export const DateTimeField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          scheduled_at: {
            type: "string",
            title: "Scheduled At",
            format: "date-time",
          },
        },
      }}
    />
  ),
};
