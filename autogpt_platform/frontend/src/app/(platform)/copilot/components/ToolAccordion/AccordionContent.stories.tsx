import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  ContentBadge,
  ContentCard,
  ContentCardDescription,
  ContentCardHeader,
  ContentCardSubtitle,
  ContentCardTitle,
  ContentCodeBlock,
  ContentGrid,
  ContentHint,
  ContentLink,
  ContentMessage,
  ContentSuggestionsList,
} from "./AccordionContent";

const meta: Meta = {
  title: "CoPilot/Content/AccordionContent",
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Building-block components used inside ToolAccordion panels to render structured content.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="max-w-[480px]">
        <Story />
      </div>
    ),
  ],
};
export default meta;

export const Grid: StoryObj = {
  render: () => (
    <ContentGrid>
      <ContentMessage>First item in the grid</ContentMessage>
      <ContentMessage>Second item in the grid</ContentMessage>
    </ContentGrid>
  ),
};

export const Card: StoryObj = {
  render: () => (
    <ContentCard>
      <ContentCardHeader>
        <ContentCardTitle>Weather Block</ContentCardTitle>
      </ContentCardHeader>
      <ContentCardSubtitle>get_weather_v2</ContentCardSubtitle>
      <ContentCardDescription>
        Fetches current weather data for a given location using the OpenWeather
        API.
      </ContentCardDescription>
    </ContentCard>
  ),
};

export const Message: StoryObj = {
  render: () => (
    <ContentMessage>
      Your agent has been created and is ready to run.
    </ContentMessage>
  ),
};

export const CodeBlock: StoryObj = {
  render: () => (
    <ContentCodeBlock>
      {JSON.stringify({ location: "London", units: "metric" }, null, 2)}
    </ContentCodeBlock>
  ),
};

export const BadgeAndLink: StoryObj = {
  render: () => (
    <div className="flex items-center gap-2">
      <ContentBadge>Required</ContentBadge>
      <ContentBadge>Optional</ContentBadge>
      <ContentLink href="https://docs.agpt.co">View Docs</ContentLink>
    </div>
  ),
};

export const Hint: StoryObj = {
  render: () => (
    <ContentHint>
      Tip: You can pass workspace:// references as input values.
    </ContentHint>
  ),
};

export const SuggestionsList: StoryObj = {
  render: () => (
    <ContentSuggestionsList
      items={[
        "Try running with different inputs",
        "Check the block documentation",
        "Ensure your API key is configured",
      ]}
    />
  ),
};
