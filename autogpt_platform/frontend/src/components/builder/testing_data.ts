// Default state data

import { BlockCategory } from "./block-menu/default/AllBlocksContent";
import { BlockListType } from "./block-menu/default/BlockMenuDefaultContent";
import { IntegrationBlockData } from "./block-menu/default/IntegrationBlocks";
import { IntegrationData } from "./block-menu/default/IntegrationList";
import { MarketplaceAgent } from "./block-menu/default/MarketplaceAgentsContent";
import { UserAgent } from "./block-menu/default/MyAgentsContent";
import { SearchItem } from "./block-menu/search-and-filter/SearchList.";

// Suggestion

export const recentSearchesData = [
  "image generator",
  "deepfake",
  "competitor analysis",
  "market research",
  "AI tools",
  "content creation",
  "data visualization",
  "automation workflow",
  "analytics dashboard",
];

// Define data for integrations
export const integrationsData = [
  {
    icon_url: "/integrations/x.png",
    name: "Twitter",
  },
  { icon_url: "/integrations/github.png", name: "Github" },
  { icon_url: "/integrations/hubspot.png", name: "Hubspot" },
  { icon_url: "/integrations/discord.png", name: "Discord" },
  { icon_url: "/integrations/medium.png", name: "Medium" },
  { icon_url: "/integrations/todoist.png", name: "Todoist" },
];

// Define data for top blocks
export const topBlocksData = [
  {
    title: "Find in Dictionary",
    description: "Enables your agent to chat with users in natural language.",
  },
  {
    title: "Web Search",
    description: "Allows your agent to search the web for information.",
  },
  {
    title: "Code Interpreter",
    description: "Helps your agent understand and execute code snippets.",
  },
  {
    title: "Data Analysis",
    description:
      "Enables your agent to analyze data and create visualizations.",
  },
  {
    title: "File Manager",
    description: "Gives your agent the ability to manage files and documents.",
  },
];

// All Blocks
export const allBlocksDataWithCategories: BlockCategory[] = [
  {
    name: "AI",
    count: 10,
    items: [
      {
        title: "Natural Language Processing",
        description:
          "Enables your agent to chat with users in natural language.",
      },
      {
        title: "Sentiment Analysis",
        description:
          "Analyzes the sentiment of user messages to respond appropriately.",
      },
      {
        title: "Text Generation",
        description:
          "Creates human-like text based on the context and inputs provided.",
      },
      {
        title: "Entity Recognition",
        description: "Identifies and extracts entities from user messages.",
      },
    ],
  },
  {
    name: "Basic",
    count: 6,
    items: [
      {
        title: "Condition",
        description: "Creates branching logic based on specific conditions.",
      },
      {
        title: "Loop",
        description: "Repeats actions until a specific condition is met.",
      },
      {
        title: "Variable",
        description:
          "Stores and manages data for use throughout your workflow.",
      },
    ],
  },
  {
    name: "Communication",
    count: 6,
    items: [
      {
        title: "Email Sender",
        description: "Sends emails to users based on triggers or conditions.",
      },
      {
        title: "SMS Notification",
        description:
          "Sends text message notifications to users' mobile devices.",
      },
      {
        title: "Webhook",
        description:
          "Integrates with external services through HTTP callbacks.",
      },
    ],
  },
];

export const actionBlocksListData: BlockListType[] = [
  {
    id: 1,
    title: "Date Input Block",
    description: "Input a date into your agent.",
  },
  {
    id: 2,
    title: "Dropdown input",
    description: "Give your users the ability to select from a dropdown menu",
  },
  {
    id: 3,
    title: "File upload",
    description: "Upload a file to your agent",
  },
  {
    id: 4,
    title: "Text input",
    description: "Allow users to select multiple options using checkboxes",
  },
];

export const inputBlocksListData: BlockListType[] = [
  {
    id: 1,
    title: "Text Field",
    description: "Collect single line text input from users.",
  },
  {
    id: 2,
    title: "Checkbox",
    description: "Allow users to select multiple options using checkboxes.",
  },
  {
    id: 3,
    title: "Radio Button",
    description: "Let users choose one option from a list of alternatives.",
  },
  {
    id: 4,
    title: "Textarea",
    description: "Collect multi-line text input from users.",
  },
  {
    id: 5,
    title: "Number Input",
    description: "Collect numerical values with optional min/max constraints.",
  },
];

export const outputBlocksListData: BlockListType[] = [
  {
    id: 1,
    title: "Display Text",
    description: "Show formatted text content to users.",
  },
  {
    id: 2,
    title: "Image Output",
    description: "Display images, charts, or visual content.",
  },
  {
    id: 3,
    title: "Table Display",
    description: "Present data in an organized tabular format.",
  },
  {
    id: 4,
    title: "PDF Generation",
    description: "Create and export data as PDF documents.",
  },
  {
    id: 5,
    title: "Status Alert",
    description: "Show success, error, or informational alerts to users.",
  },
];

export const integrationsListData: IntegrationData[] = [
  {
    title: "Twitter",
    icon_url: "/integrations/x.png",
    description:
      "All twitter blocks, It has everthing to interact with twitter",
    number_of_blocks: 4,
  },
  {
    title: "Discord",
    icon_url: "/integrations/discord.png",
    description:
      "All Discord blocks, It has everthing to interact with discord",
    number_of_blocks: 4,
  },
  {
    title: "Github",
    icon_url: "/integrations/github.png",
    description: "All Github blocks, It has everthing to interact with github",
    number_of_blocks: 4,
  },
  {
    title: "Hubspot",
    icon_url: "/integrations/hubspot.png",
    description:
      "All Hubspot blocks, It has everthing to interact with Hubspot",
    number_of_blocks: 2,
  },
  {
    title: "Medium",
    icon_url: "/integrations/medium.png",
    description: "All Medium blocks, It has everything to interact with Medium",
    number_of_blocks: 4,
  },
  {
    title: "Todoist",
    icon_url: "/integrations/todoist.png",
    description:
      "All Todoist blocks, It has everything to interact with Todoist",
    number_of_blocks: 4,
  },
];

export const marketplaceAgentData: MarketplaceAgent[] = [
  {
    id: 1,
    title: "turtle test",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 1000,
  },
  {
    id: 2,
    title: "turtle test 1",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 1324,
  },
  {
    id: 3,
    title: "turtle test 2",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 10030,
  },
  {
    id: 4,
    title: "turtle test 3",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 324,
  },
  {
    id: 5,
    title: "turtle test",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 4345,
  },
  {
    id: 6,
    title: "turtle test",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 324,
  },
  {
    id: 7,
    title: "turtle test 3",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 324,
  },
  {
    id: 8,
    title: "turtle test",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 4345,
  },
  {
    id: 9,
    title: "turtle test",
    image_url: "/placeholder.png",
    creator_name: "Autogpt",
    number_of_runs: 324,
  },
];

export const myAgentData: UserAgent[] = [
  {
    id: 1,
    title: "My Agent 1",
    edited_time: "23rd April",
    version: 3,
    image_url: "/placeholder.png",
  },
  {
    id: 2,
    title: "My Agent 2",
    edited_time: "21st April",
    version: 4,
    image_url: "/placeholder.png",
  },
  {
    id: 3,
    title: "My Agent 3",
    edited_time: "23rd May",
    version: 7,
    image_url: "/placeholder.png",
  },
  {
    id: 4,
    title: "My Agent 4",
    edited_time: "23rd April",
    version: 3,
    image_url: "/placeholder.png",
  },
  {
    id: 5,
    title: "My Agent 5",
    edited_time: "23rd April",
    version: 3,
    image_url: "/placeholder.png",
  },
  {
    id: 6,
    title: "My Agent 6",
    edited_time: "23rd April",
    version: 3,
    image_url: "/placeholder.png",
  },
];

export const integrationBlocksData: Record<string, IntegrationBlockData[]> = {
  Twitter: [
    {
      title: "Twitter Blocks: Post tweet",
      description: "Post tweet on twitter",
      icon_url: "/integrations/x.png",
    },
    {
      title: "Twitter Blocks: Delete tweet",
      description: "Delete tweet on twitter",
      icon_url: "/integrations/x.png",
    },
    {
      title: "Twitter Blocks: Update tweet",
      description: "Update tweet on twitter",
      icon_url: "/integrations/x.png",
    },
    {
      title: "Twitter Blocks: Retweet tweet",
      description: "Retweet tweet on twitter",
      icon_url: "/integrations/x.png",
    },
  ],
  Discord: [
    {
      title: "Discord Blocks: Create",
      description: "Create message on discord",
      icon_url: "/integrations/discord.png",
    },
    {
      title: "Discord Blocks: Delete",
      description: "Delete message on discord",
      icon_url: "/integrations/discord.png",
    },
    {
      title: "Discord Blocks: Update",
      description: "Update message on discord",
      icon_url: "/integrations/discord.png",
    },
    {
      title: "Discord Blocks: Read",
      description: "Read message on discord",
      icon_url: "/integrations/discord.png",
    },
  ],
  Github: [
    {
      title: "Github Blocks: Create PR",
      description: "Create pull request on github",
      icon_url: "/integrations/github.png",
    },
    {
      title: "Github Blocks: Merge PR",
      description: "Merge pull request on github",
      icon_url: "/integrations/github.png",
    },
  ],
  Hubspot: [
    {
      title: "Hubspot Blocks: Create Contact",
      description: "Create contact on hubspot",
      icon_url: "/integrations/hubspot.png",
    },
    {
      title: "Hubspot Blocks: Update Contact",
      description: "Update contact on hubspot",
      icon_url: "/integrations/hubspot.png",
    },
  ],
  Medium: [
    {
      title: "Medium Blocks: Post Article",
      description: "Post article on medium",
      icon_url: "/integrations/medium.png",
    },
    {
      title: "Medium Blocks: Delete Article",
      description: "Delete article on medium",
      icon_url: "/integrations/medium.png",
    },
  ],
  Todoist: [
    {
      title: "Todoist Blocks: Create Task",
      description: "Create task on todoist",
      icon_url: "/integrations/todoist.png",
    },
    {
      title: "Todoist Blocks: Complete Task",
      description: "Complete task on todoist",
      icon_url: "/integrations/todoist.png",
    },
  ],
};

export const searchingData: SearchItem[] = [
  {
    type: "marketing_agent" as const,
    title: marketplaceAgentData[0].title,
    image_url: marketplaceAgentData[0].image_url,
    creator_name: marketplaceAgentData[0].creator_name,
    number_of_runs: marketplaceAgentData[0].number_of_runs,
  },

  {
    type: "ai",
    title: "Natural Language Processing",
    description: "Enables your agent to chat with users in natural language.",
    ai_name: "Claude 3.5 Sonnet",
  },

  {
    type: "integration_block" as const,
    title: integrationsListData[0].title,
    description: integrationsListData[0].description,
    icon_url: integrationsListData[0].icon_url,
    number_of_blocks: integrationsListData[0].number_of_blocks,
  },

  {
    type: "marketing_agent" as const,
    title: marketplaceAgentData[1].title,
    image_url: marketplaceAgentData[1].image_url,
    creator_name: marketplaceAgentData[1].creator_name,
    number_of_runs: marketplaceAgentData[1].number_of_runs,
  },

  {
    type: "block" as const,
    title: topBlocksData[0].title,
    description: topBlocksData[0].description,
  },

  {
    type: "my_agent" as const,
    title: myAgentData[0].title,
    image_url: myAgentData[0].image_url,
    edited_time: myAgentData[0].edited_time,
    version: myAgentData[0].version,
  },

  {
    type: "ai",
    title: "Sentiment Analysis",
    description:
      "Analyzes the sentiment of user messages to respond appropriately.",
    ai_name: "Claude 3.5 Sonnet",
  },

  {
    type: "block" as const,
    title: topBlocksData[1].title,
    description: topBlocksData[1].description,
  },

  {
    type: "marketing_agent" as const,
    title: marketplaceAgentData[2].title,
    image_url: marketplaceAgentData[2].image_url,
    creator_name: marketplaceAgentData[2].creator_name,
    number_of_runs: marketplaceAgentData[2].number_of_runs,
  },

  {
    type: "integration_block" as const,
    title: integrationsListData[1].title,
    description: integrationsListData[1].description,
    icon_url: integrationsListData[1].icon_url,
    number_of_blocks: integrationsListData[1].number_of_blocks,
  },

  {
    type: "my_agent" as const,
    title: myAgentData[1].title,
    image_url: myAgentData[1].image_url,
    edited_time: myAgentData[1].edited_time,
    version: myAgentData[1].version,
  },
];
