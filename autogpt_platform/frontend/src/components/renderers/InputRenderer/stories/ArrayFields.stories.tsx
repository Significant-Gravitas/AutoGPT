import type { Meta, StoryObj } from "@storybook/nextjs";
import { FormRendererStory, storyDecorator } from "./FormRendererStoryWrapper";

const meta: Meta = {
  title: "Renderers/FormRenderer/Array Fields",
  tags: ["autodocs"],
  decorators: [storyDecorator],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Array field types: list[str], list[int], list[Enum], list[bool], and list[object].",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const ListOfStrings: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          tags: {
            type: "array",
            title: "Tags",
            items: { type: "string" },
            description: "list[str] - A list of text items",
          },
        },
      }}
      initialValues={{ tags: ["tag1", "tag2"] }}
    />
  ),
};

export const ListOfIntegers: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          numbers: {
            type: "array",
            title: "Numbers",
            items: { type: "integer" },
            description: "list[int] - A list of integers",
          },
        },
      }}
      initialValues={{ numbers: [1, 2, 3] }}
    />
  ),
};

export const ListOfEnums: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          formats: {
            type: "array",
            title: "Formats",
            items: {
              type: "string",
              enum: ["markdown", "html", "screenshot", "rawHtml", "links"],
            },
            description: "list[Enum] - e.g. Firecrawl ScrapeFormat",
          },
        },
      }}
      initialValues={{ formats: ["markdown", "screenshot"] }}
    />
  ),
};

export const ListOfBooleans: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          flags: {
            type: "array",
            title: "Flags",
            items: { type: "boolean" },
            description: "list[bool] - A list of boolean flags",
          },
        },
      }}
      initialValues={{ flags: [true, false] }}
    />
  ),
};

export const ListOfObjects: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          headers: {
            type: "array",
            title: "Headers",
            items: {
              type: "object",
              properties: {
                key: { type: "string", title: "Key" },
                value: { type: "string", title: "Value" },
              },
            },
            description: "list[dict] - Key-value pairs",
          },
        },
      }}
      initialValues={{
        headers: [{ key: "Authorization", value: "Bearer token" }],
      }}
    />
  ),
};
