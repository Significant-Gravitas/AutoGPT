import type { Meta, StoryObj } from "@storybook/nextjs";
import { ErrorCard } from "./ErrorCard";

const meta: Meta<typeof ErrorCard> = {
  title: "Molecules/ErrorCard",
  component: ErrorCard,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: `
## ErrorCard Component

A reusable error card component that handles API query responses gracefully with elegant styling and user-friendly messaging.

### ✨ Features

- **Purple gradient border** - Elegant gradient from purple to light pastel purple
- **Smart error handling** - Automatically detects HTTP errors vs response errors  
- **User-friendly messages** - Non-technical, funny error messages for HTTP errors
- **Loading state** - Supports custom loading slot or default spinner
- **Action buttons** - Sentry error reporting with toast notifications and Discord help link
- **Phosphor icons** - Uses phosphor icons throughout
- **TypeScript support** - Full TypeScript interface support

### 🎯 Magic Usage Pattern

Just pass your API hook results directly:

\`\`\`tsx
<ErrorCard
  isSuccess={isSuccess}
  responseError={error || undefined}
  httpError={response?.status !== 200 ? { 
    status: response?.status,
    statusText: "Request failed" 
  } : undefined}
  context="agent data"
  onRetry={refetch}
/>
\`\`\`

The component will automatically:
1. Show loading spinner if not successful and no errors
2. Show custom loading slot if provided  
3. Handle HTTP errors with friendly messages
4. Handle response errors with technical details
5. Report errors to Sentry with comprehensive context
6. Show toast notifications for error reporting feedback
7. Provide retry and Discord help options
8. Hide itself if successful and no errors

### 🎭 User-Friendly HTTP Error Messages

- **500+**: "Our servers are having a bit of a moment 🤖"
- **404**: "We couldn't find what you're looking for. It might have wandered off somewhere! 🔍"  
- **403**: "You don't have permission to access this. Maybe you need to sign in again? 🔐"
- **429**: "Whoa there, speed racer! You're making requests too quickly. Take a breather and try again. ⏱️"
- **400+**: "Something's not quite right with your request. Double-check and try again! ✨"
        `,
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    isSuccess: {
      control: "boolean",
      description: "Whether the API request was successful",
      table: {
        defaultValue: { summary: "false" },
      },
    },
    responseError: {
      control: "object",
      description: "Error object from API response (validation errors, etc.)",
    },
    httpError: {
      control: "object",
      description: "HTTP error object with status code and message",
    },
    context: {
      control: "text",
      description: "Context for the error message (e.g., 'user data', 'agent')",
      table: {
        defaultValue: { summary: '"data"' },
      },
    },
    onRetry: {
      control: false,
      description:
        "Callback function for retry button (button only shows if provided)",
    },
    className: {
      control: "text",
      description: "Additional CSS classes to apply",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Response errors are typically validation errors from the API.
 * They show the technical error message in a code block for debugging.
 */
export const ResponseError: Story = {
  args: {
    isSuccess: false,
    responseError: {
      detail: [{ msg: "Invalid authentication credentials provided" }],
    },
    context: "user data",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * HTTP 500+ errors get a friendly, non-technical message about server issues.
 */
export const HttpError500: Story = {
  args: {
    isSuccess: false,
    httpError: {
      status: 500,
      statusText: "Internal Server Error",
    },
    context: "agent data",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * HTTP 404 errors get a playful message about missing resources.
 */
export const HttpError404: Story = {
  args: {
    isSuccess: false,
    httpError: {
      status: 404,
      statusText: "Not Found",
    },
    context: "agent data",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * HTTP 429 errors get a humorous rate limiting message.
 */
export const HttpError429: Story = {
  args: {
    isSuccess: false,
    httpError: {
      status: 429,
      statusText: "Too Many Requests",
    },
    context: "API data",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * HTTP 403 errors suggest re-authentication.
 */
export const HttpError403: Story = {
  args: {
    isSuccess: false,
    httpError: {
      status: 403,
      statusText: "Forbidden",
    },
    context: "user profile",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * Default loading state shows a spinning phosphor icon.
 */
export const LoadingState: Story = {
  args: {
    isSuccess: false,
  },
};

/**
 * Response errors can also have string error details instead of arrays.
 */
export const StringErrorDetail: Story = {
  args: {
    isSuccess: false,
    responseError: {
      detail: "Something went wrong with the database connection",
    },
    context: "database",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * If no onRetry callback is provided, the retry button won't appear.
 */
export const NoRetryButton: Story = {
  args: {
    isSuccess: false,
    responseError: {
      message: "This error cannot be retried",
    },
    context: "configuration",
    // No onRetry prop - button won't show
  },
};

/**
 * Response errors with just a message property.
 */
export const SimpleMessage: Story = {
  args: {
    isSuccess: false,
    responseError: {
      message: "User session has expired",
    },
    context: "authentication",
    onRetry: () => alert("Retry clicked!"),
  },
};

/**
 * Typical usage pattern with React Query or similar data fetching hooks.
 */
export const TypicalUsage: Story = {
  args: {
    isSuccess: false,
    responseError: {
      detail: [{ msg: "Agent not found in library" }],
    },
    context: "agent",
    onRetry: () => alert("This would typically call refetch() or similar"),
  },
  parameters: {
    docs: {
      description: {
        story: `
This shows how you'd typically use ErrorCard with a data fetching hook:

\`\`\`tsx
function MyComponent() {
  const { data: response, isSuccess, error } = useApiHook();
  
  return (
    <ErrorCard
      isSuccess={isSuccess}
      responseError={error || undefined}
      httpError={response?.status !== 200 ? {
        status: response?.status,
        statusText: "Request failed"
      } : undefined}
      context="agent"
      onRetry={() => window.location.reload()}
    />
  );
}
\`\`\`
        `,
      },
    },
  },
};
