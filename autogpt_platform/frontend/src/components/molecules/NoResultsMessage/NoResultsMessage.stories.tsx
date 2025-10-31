import type { Meta, StoryObj } from "@storybook/nextjs";
import { NoResultsMessage } from "./NoResultsMessage";

const meta = {
  title: "Molecules/NoResultsMessage",
  component: NoResultsMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof NoResultsMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    message:
      "No agents found matching 'crypto mining'. Try different keywords or browse the marketplace.",
    suggestions: [
      "Try more general terms",
      "Browse categories in the marketplace",
      "Check spelling",
    ],
  },
};

export const NoSuggestions: Story = {
  args: {
    message: "No results found for your search. Please try a different query.",
    suggestions: [],
  },
};

export const LongMessage: Story = {
  args: {
    message:
      "We couldn't find any agents matching your search criteria. This could be because the agent you're looking for doesn't exist yet, or you might need to adjust your search terms to be more specific or more general depending on what you're trying to find.",
    suggestions: [
      "Try using different keywords",
      "Browse all available agents in the marketplace",
      "Check your spelling and try again",
      "Consider creating your own agent for this use case",
    ],
  },
};

export const ShortMessage: Story = {
  args: {
    message: "No results.",
    suggestions: ["Try again"],
  },
};

export const ManySuggestions: Story = {
  args: {
    message: "No agents found matching your criteria.",
    suggestions: [
      "Use broader search terms",
      "Try searching by category",
      "Check the spelling of your keywords",
      "Browse the full marketplace",
      "Consider synonyms or related terms",
      "Filter by specific features",
    ],
  },
};
