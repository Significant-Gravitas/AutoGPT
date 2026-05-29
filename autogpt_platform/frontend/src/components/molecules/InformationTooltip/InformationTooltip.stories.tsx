import type { Meta, StoryObj } from "@storybook/nextjs";
import { InformationTooltip } from "./InformationTooltip";

const meta: Meta<typeof InformationTooltip> = {
  title: "Molecules/InformationTooltip",
  tags: ["autodocs"],
  component: InformationTooltip,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "InformationTooltip component displays contextual help information with markdown support. Shows an info icon that reveals a tooltip on hover with formatted content.",
      },
    },
  },
  argTypes: {
    description: {
      control: "text",
      description: "Markdown content to display in the tooltip",
    },
  },
  args: {
    description: "This is helpful information about the feature.",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    description: "This is helpful information about the feature.",
  },
};

export const WithMarkdown: Story = {
  args: {
    description: `This tooltip supports **markdown formatting**:

- Bullet points
- *Italic text*
- **Bold text**

You can also include [links](https://example.com) that open in new tabs.`,
  },
};

export const WithLink: Story = {
  args: {
    description:
      "Visit our [documentation](https://docs.example.com) for more details.",
  },
};

export const LongDescription: Story = {
  args: {
    description: `This is a longer description that demonstrates how the tooltip handles extended content. The tooltip has a maximum width and will wrap text appropriately to ensure readability while maintaining a clean appearance.

It can include multiple paragraphs and various formatting elements to provide comprehensive help information to users.`,
  },
};

export const WithCode: Story = {
  args: {
    description:
      "Use the `API_KEY` environment variable to configure authentication.",
  },
};

export const NoDescription: Story = {
  args: {
    description: undefined,
  },
  parameters: {
    docs: {
      description: {
        story:
          "When no description is provided, the component returns null and doesn't render anything.",
      },
    },
  },
};

export const EmptyDescription: Story = {
  args: {
    description: "",
  },
  parameters: {
    docs: {
      description: {
        story:
          "When an empty string is provided, the component returns null and doesn't render anything.",
      },
    },
  },
};

export const MultipleTooltips: Story = {
  render: renderMultipleTooltips,
  parameters: {
    docs: {
      description: {
        story: "Multiple tooltips can be used together in a form or interface.",
      },
    },
  },
};

function renderMultipleTooltips() {
  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex items-center gap-2">
        <span>API Configuration</span>
        <InformationTooltip description="Configure your API settings here. Make sure to use a valid API key." />
      </div>
      <div className="flex items-center gap-2">
        <span>Webhook URL</span>
        <InformationTooltip description="The webhook URL where notifications will be sent. Must be a valid HTTPS endpoint." />
      </div>
      <div className="flex items-center gap-2">
        <span>Rate Limiting</span>
        <InformationTooltip description="Control the number of requests per minute. Higher values may impact performance." />
      </div>
    </div>
  );
}
