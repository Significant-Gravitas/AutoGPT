import { StoryCode } from "@/stories/helpers/StoryCode";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { Text, textVariants } from "./Text";

const meta: Meta<typeof Text> = {
  title: "Design System/Atoms/Text",
  component: Text,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
    controls: { hideNoControlsWarning: true },
    docs: {
      description: {
        component:
          "A flexible Text component that supports all typography variants from our design system. Uses Poppins for headings and Geist Sans for body text.",
      },
      source: {
        state: "open",
      },
    },
  },
  argTypes: {
    variant: {
      control: { type: "select" },
      options: textVariants,
      description: "Typography variant to apply",
    },
    as: {
      control: { type: "select" },
      options: ["h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "div", "code"],
      description: "HTML element to render as",
    },
    children: {
      control: "text",
      description: "Text content",
    },
  },
};

export default meta;
type Story = StoryObj<typeof Text>;

//=============================================================================
// All Variants Overview
//=============================================================================
export function AllVariants() {
  return (
    <div className="space-y-8">
      {/* Headings */}
      <div className="mb-19 mb-20 space-y-6">
        <h2 className="mb-4 border-b border-border pb-2 text-xl text-zinc-500">
          Headings
        </h2>
        <Text variant="h1">Heading 1</Text>
        <Text variant="h2">Heading 2</Text>
        <Text variant="h3">Heading 3</Text>
        <Text variant="h4">Heading 4</Text>
        <StoryCode
          code={`<Text variant="h1">Heading 1</Text>
<Text variant="h2">Heading 2</Text>
<Text variant="h3">Heading 3</Text>
<Text variant="h4">Heading 4</Text>`}
        />
      </div>
      {/* Body Text */}
      <h2 className="mb-4 border-b border-border pb-2 text-xl text-zinc-500">
        Body Text
      </h2>
      <Text variant="lead">Lead</Text>
      <StoryCode code={`<Text variant="lead">Lead</Text>`} />
      <div className="flex flex-row gap-8">
        <Text variant="large">Large</Text>
        <Text variant="large-medium">Large Medium</Text>
        <Text variant="large-semibold">Large Semibold</Text>
      </div>
      <StoryCode
        code={`<Text variant="large">Large</Text>
<Text variant="large-medium">Large Medium</Text>
<Text variant="large-semibold">Large Semibold</Text>`}
      />
      <div className="flex flex-row gap-8">
        <Text variant="body">Body</Text>
        <Text variant="body-medium">Body Medium</Text>
      </div>
      <StoryCode
        code={`<Text variant="body">Body</Text>
<Text variant="body-medium">Body Medium</Text>`}
      />
      <div className="flex flex-row gap-8">
        <Text variant="small">Small</Text>
        <Text variant="small-medium">Small Medium</Text>
      </div>
      <StoryCode
        code={`<Text variant="small">Small</Text>
<Text variant="small-medium">Small Medium</Text>`}
      />
      <Text variant="subtle">Subtle</Text>
      <StoryCode code={`<Text variant="subtle">Subtle</Text>`} />
    </div>
  );
}

//=============================================================================
// Headings Only
//=============================================================================
export function Headings() {
  return (
    <div className="space-y-8">
      <Text variant="h1">Heading 1</Text>
      <Text variant="h2">Heading 2</Text>
      <Text variant="h3">Heading 3</Text>
      <Text variant="h4">Heading 4</Text>
    </div>
  );
}

//=============================================================================
// Body Text Only
//=============================================================================
export function BodyText() {
  return (
    <div className="space-y-8">
      <Text variant="lead">Lead</Text>
      <Text variant="large">Large</Text>
      <Text variant="large-medium">Large Medium</Text>
      <Text variant="large-semibold">Large Semibold</Text>
      <Text variant="body">Body</Text>
      <Text variant="body-medium">Body Medium</Text>
      <Text variant="small">Small</Text>
      <Text variant="small-medium">Small Medium</Text>
      <Text variant="subtle">Subtle</Text>
    </div>
  );
}

//=============================================================================
// Interactive Playground
//=============================================================================
export const Playground: Story = {
  args: {
    variant: "body",
    children:
      "Edit this text and try different variants via the controls below",
  },
  parameters: {
    controls: { include: ["variant", "as", "children"] },
  },
  render: (args) => (
    <div className="space-y-8">
      <Text {...args} />
    </div>
  ),
};

//=============================================================================
// Custom Element
//=============================================================================
export function CustomElement() {
  return (
    <div className="space-y-8">
      <Text variant="h3" as="div">
        H3 size rendered as div
      </Text>
      <Text variant="body" as="h2">
        Body size rendered as h2
      </Text>
    </div>
  );
}
