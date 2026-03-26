import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { FormRenderer } from "../FormRenderer";
import type { RJSFSchema } from "@rjsf/utils";
import type { ExtendedFormContextType } from "../types";

const defaultFormContext: ExtendedFormContextType = {
  nodeId: "story-node",
  showHandles: false,
  size: "medium",
  showOptionalToggle: true,
};

function FormRendererStory({
  jsonSchema,
  initialValues,
}: {
  jsonSchema: RJSFSchema;
  initialValues?: Record<string, unknown>;
}) {
  const [formData, setFormData] = useState(initialValues ?? {});

  return (
    <div className="w-[400px]">
      <FormRenderer
        jsonSchema={jsonSchema}
        handleChange={(e) => setFormData(e.formData)}
        uiSchema={{}}
        initialValues={formData}
        formContext={defaultFormContext}
      />
    </div>
  );
}

const meta: Meta = {
  title: "Renderers/FormRenderer",
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "FormRenderer renders JSON Schema-based forms used in the node builder. These stories showcase every field type the backend can produce.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// --- Primitive Types ---

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

// --- Enum / Select ---

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

// --- Array Types ---

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

// --- Object / Nested Types ---

export const ObjectField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          config: {
            type: "object",
            title: "Config",
            properties: {
              host: { type: "string", title: "Host" },
              port: { type: "integer", title: "Port" },
              ssl: { type: "boolean", title: "SSL" },
            },
            description: "A nested object with multiple fields",
          },
        },
      }}
    />
  ),
};

export const NestedObjectWithEnum: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          settings: {
            type: "object",
            title: "Settings",
            properties: {
              mode: {
                type: "string",
                title: "Mode",
                enum: ["fast", "balanced", "quality"],
              },
              max_retries: { type: "integer", title: "Max Retries" },
              verbose: { type: "boolean", title: "Verbose" },
            },
          },
        },
      }}
    />
  ),
};

// --- Optional / AnyOf ---

export const OptionalString: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          nickname: {
            anyOf: [{ type: "string" }, { type: "null" }],
            title: "Nickname",
            description: "Optional[str] - can be a string or null",
          },
        },
      }}
    />
  ),
};

export const OptionalInteger: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          max_tokens: {
            anyOf: [{ type: "integer" }, { type: "null" }],
            title: "Max Tokens",
            description: "Optional[int] - can be an integer or null",
          },
        },
      }}
    />
  ),
};

// --- Union / AnyOf (multiple types) ---

export const UnionStringOrInteger: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        properties: {
          value: {
            anyOf: [{ type: "string" }, { type: "integer" }],
            title: "Value",
            description: "str | int - Union of string and integer",
          },
        },
      }}
    />
  ),
};

// --- TwitterGetUserBlock (exact schema from model_json_schema(), minus credentials) ---
// Includes: oneOf discriminated union (identifier), anyOf multi-select (expansions, tweet_fields, user_fields)

export const TwitterGetUserBlock: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={
        {
          type: "object",
          $defs: {
            UserId: {
              type: "object",
              title: "UserId",
              properties: {
                discriminator: {
                  const: "user_id",
                  title: "Discriminator",
                  type: "string",
                },
                user_id: {
                  advanced: true,
                  default: "",
                  description: "The ID of the user to lookup",
                  secret: false,
                  title: "User Id",
                  type: "string",
                },
              },
              required: ["discriminator"],
            },
            Username: {
              type: "object",
              title: "Username",
              properties: {
                discriminator: {
                  const: "username",
                  title: "Discriminator",
                  type: "string",
                },
                username: {
                  advanced: true,
                  default: "",
                  description: "The Twitter username (handle) of the user",
                  secret: false,
                  title: "Username",
                  type: "string",
                },
              },
              required: ["discriminator"],
            },
            UserExpansionsFilter: {
              type: "object",
              title: "UserExpansionsFilter",
              properties: {
                pinned_tweet_id: {
                  default: false,
                  title: "Pinned Tweet Id",
                  type: "boolean",
                },
              },
            },
            TweetFieldsFilter: {
              type: "object",
              title: "TweetFieldsFilter",
              properties: {
                Tweet_Attachments: {
                  default: false,
                  title: "Tweet Attachments",
                  type: "boolean",
                },
                Author_ID: {
                  default: false,
                  title: "Author Id",
                  type: "boolean",
                },
                Context_Annotations: {
                  default: false,
                  title: "Context Annotations",
                  type: "boolean",
                },
                Conversation_ID: {
                  default: false,
                  title: "Conversation Id",
                  type: "boolean",
                },
                Creation_Time: {
                  default: false,
                  title: "Creation Time",
                  type: "boolean",
                },
                Edit_Controls: {
                  default: false,
                  title: "Edit Controls",
                  type: "boolean",
                },
                Tweet_Entities: {
                  default: false,
                  title: "Tweet Entities",
                  type: "boolean",
                },
                Geographic_Location: {
                  default: false,
                  title: "Geographic Location",
                  type: "boolean",
                },
                Tweet_ID: {
                  default: false,
                  title: "Tweet Id",
                  type: "boolean",
                },
                Reply_To_User_ID: {
                  default: false,
                  title: "Reply To User Id",
                  type: "boolean",
                },
                Language: {
                  default: false,
                  title: "Language",
                  type: "boolean",
                },
                Public_Metrics: {
                  default: false,
                  title: "Public Metrics",
                  type: "boolean",
                },
                Sensitive_Content_Flag: {
                  default: false,
                  title: "Sensitive Content Flag",
                  type: "boolean",
                },
                Referenced_Tweets: {
                  default: false,
                  title: "Referenced Tweets",
                  type: "boolean",
                },
                Reply_Settings: {
                  default: false,
                  title: "Reply Settings",
                  type: "boolean",
                },
                Tweet_Source: {
                  default: false,
                  title: "Tweet Source",
                  type: "boolean",
                },
                Tweet_Text: {
                  default: false,
                  title: "Tweet Text",
                  type: "boolean",
                },
                Withheld_Content: {
                  default: false,
                  title: "Withheld Content",
                  type: "boolean",
                },
              },
            },
            TweetUserFieldsFilter: {
              type: "object",
              title: "TweetUserFieldsFilter",
              properties: {
                Account_Creation_Date: {
                  default: false,
                  title: "Account Creation Date",
                  type: "boolean",
                },
                User_Bio: {
                  default: false,
                  title: "User Bio",
                  type: "boolean",
                },
                User_Entities: {
                  default: false,
                  title: "User Entities",
                  type: "boolean",
                },
                User_ID: {
                  default: false,
                  title: "User Id",
                  type: "boolean",
                },
                User_Location: {
                  default: false,
                  title: "User Location",
                  type: "boolean",
                },
                Latest_Tweet_ID: {
                  default: false,
                  title: "Latest Tweet Id",
                  type: "boolean",
                },
                Display_Name: {
                  default: false,
                  title: "Display Name",
                  type: "boolean",
                },
                Pinned_Tweet_ID: {
                  default: false,
                  title: "Pinned Tweet Id",
                  type: "boolean",
                },
                Profile_Picture_URL: {
                  default: false,
                  title: "Profile Picture Url",
                  type: "boolean",
                },
                Is_Protected_Account: {
                  default: false,
                  title: "Is Protected Account",
                  type: "boolean",
                },
                Account_Statistics: {
                  default: false,
                  title: "Account Statistics",
                  type: "boolean",
                },
                Profile_URL: {
                  default: false,
                  title: "Profile Url",
                  type: "boolean",
                },
                Username: {
                  default: false,
                  title: "Username",
                  type: "boolean",
                },
                Is_Verified: {
                  default: false,
                  title: "Is Verified",
                  type: "boolean",
                },
                Verification_Type: {
                  default: false,
                  title: "Verification Type",
                  type: "boolean",
                },
                Content_Withholding_Info: {
                  default: false,
                  title: "Content Withholding Info",
                  type: "boolean",
                },
              },
            },
          },
          required: ["identifier"],
          properties: {
            identifier: {
              advanced: false,
              description:
                "Choose whether to identify the user by their unique Twitter ID or by their username",
              discriminator: {
                mapping: {
                  user_id: "#/$defs/UserId",
                  username: "#/$defs/Username",
                },
                propertyName: "discriminator",
              },
              oneOf: [{ $ref: "#/$defs/UserId" }, { $ref: "#/$defs/Username" }],
              secret: false,
              title: "Identifier",
            },
            expansions: {
              advanced: true,
              anyOf: [
                { $ref: "#/$defs/UserExpansionsFilter" },
                { type: "null" },
              ],
              default: null,
              description:
                "Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet.",
              placeholder: "Select extra user information to include",
              secret: false,
            },
            tweet_fields: {
              advanced: true,
              anyOf: [{ $ref: "#/$defs/TweetFieldsFilter" }, { type: "null" }],
              default: null,
              description:
                "Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above.",
              placeholder: "Choose what details to see in pinned tweets",
              secret: false,
            },
            user_fields: {
              advanced: true,
              anyOf: [
                { $ref: "#/$defs/TweetUserFieldsFilter" },
                { type: "null" },
              ],
              default: null,
              description:
                "Select what user information you want to see, like username, bio, profile picture, etc.",
              placeholder: "Choose what user details you want to see",
              secret: false,
            },
          },
        } as RJSFSchema
      }
      initialValues={{
        identifier: { discriminator: "user_id", user_id: "" },
      }}
    />
  ),
};

// --- Multi-select (all-boolean object, exact Twitter TweetFieldsFilter schema) ---

export const MultiSelectField: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={
        {
          type: "object",
          $defs: {
            TweetFieldsFilter: {
              type: "object",
              title: "TweetFieldsFilter",
              properties: {
                Tweet_Attachments: {
                  default: false,
                  title: "Tweet Attachments",
                  type: "boolean",
                },
                Author_ID: {
                  default: false,
                  title: "Author Id",
                  type: "boolean",
                },
                Context_Annotations: {
                  default: false,
                  title: "Context Annotations",
                  type: "boolean",
                },
                Conversation_ID: {
                  default: false,
                  title: "Conversation Id",
                  type: "boolean",
                },
                Creation_Time: {
                  default: false,
                  title: "Creation Time",
                  type: "boolean",
                },
                Tweet_Entities: {
                  default: false,
                  title: "Tweet Entities",
                  type: "boolean",
                },
                Language: {
                  default: false,
                  title: "Language",
                  type: "boolean",
                },
                Public_Metrics: {
                  default: false,
                  title: "Public Metrics",
                  type: "boolean",
                },
                Tweet_Text: {
                  default: false,
                  title: "Tweet Text",
                  type: "boolean",
                },
              },
            },
          },
          properties: {
            tweet_fields: {
              anyOf: [{ $ref: "#/$defs/TweetFieldsFilter" }, { type: "null" }],
              default: null,
              description: "Select what tweet information you want to see.",
              placeholder: "Choose what details to see in tweets",
            },
          },
        } as RJSFSchema
      }
    />
  ),
};

// --- Required vs Optional fields ---

export const RequiredFields: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        required: ["email", "role"],
        properties: {
          email: { type: "string", title: "Email" },
          role: {
            type: "string",
            title: "Role",
            enum: ["admin", "editor", "viewer"],
          },
          bio: { type: "string", title: "Bio" },
        },
      }}
    />
  ),
};

// --- List of Objects ---

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

// --- Complex / Kitchen Sink ---

export const KitchenSink: Story = {
  render: () => (
    <FormRendererStory
      jsonSchema={{
        type: "object",
        required: ["url", "method"],
        properties: {
          url: {
            type: "string",
            title: "URL",
            description: "The target URL",
          },
          method: {
            type: "string",
            title: "Method",
            enum: ["GET", "POST", "PUT", "DELETE", "PATCH"],
          },
          timeout: {
            type: "number",
            title: "Timeout (seconds)",
            description: "Request timeout",
          },
          follow_redirects: {
            type: "boolean",
            title: "Follow Redirects",
          },
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
          },
          body_format: {
            type: "string",
            title: "Body Format",
            enum: ["json", "form", "raw", "none"],
          },
          tags: {
            type: "array",
            title: "Tags",
            items: { type: "string" },
          },
          auth: {
            anyOf: [
              {
                type: "object",
                title: "Bearer Token",
                properties: {
                  token: { type: "string", title: "Token" },
                },
              },
              { type: "null" },
            ],
            title: "Authentication",
          },
        },
      }}
      initialValues={{
        method: "GET",
        follow_redirects: true,
        body_format: "json",
      }}
    />
  ),
};
