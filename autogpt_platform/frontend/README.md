This is the frontend for AutoGPT's next generation

## 🧢 Getting Started

This project uses [**pnpm**](https://pnpm.io/) as the package manager via **corepack**. [Corepack](https://github.com/nodejs/corepack) is a Node.js tool that automatically manages package managers without requiring global installations.

### Prerequisites

Make sure you have Node.js 16.10+ installed. Corepack is included with Node.js by default.

### ⚠️ Migrating from yarn

> This project was previously using yarn1, make sure to clean up the old files if you set it up previously with yarn:
>
> ```bash
> rm -f yarn.lock && rm -rf node_modules
> ```
>
> Then follow the setup steps below.

## Setup

### 1. **Enable corepack** (run this once on your system):

```bash
corepack enable
```

This enables corepack to automatically manage pnpm based on the `packageManager` field in `package.json`.

### 2. **Install dependencies**:

```bash
pnpm i
```

### 3. **Start the development server**:

#### Running the Front-end & Back-end separately

We recommend this approach if you are doing active development on the project. First spin up the Back-end:

```bash
# on `autogpt_platform`
docker compose --profile local up deps_backend -d
# on `autogpt_platform/backend`
poetry run app
```

Then start the Front-end:

```bash
# on `autogpt_platform/frontend`
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result. If the server starts on `http://localhost:3001` it means the Front-end is already running via Docker. You have to kill the container then or do `docker compose down`.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

#### Running both the Front-end and Back-end via Docker

If you run:

```bash
# on `autogpt_platform`
docker compose up -d
```

It will spin up the Back-end and Front-end via Docker. The Front-end will start on port `3000`. This might not be
what you want when actively contributing to the Front-end as you won't have direct/easy access to the Next.js dev server.

### Subsequent Runs

For subsequent development sessions, you only need to run:

```bash
pnpm dev
```

Every time a new Front-end dependency is added by you or others, you will need to run `pnpm i` to install the new dependencies.

### Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm start` - Start production server
- `pnpm lint` - Run ESLint and Prettier checks
- `pnpm format` - Format code with Prettier
- `pnpm types` - Run TypeScript type checking
- `pnpm test` - Run Playwright tests
- `pnpm test-ui` - Run Playwright tests with UI
- `pnpm fetch:openapi` - Fetch OpenAPI spec from backend
- `pnpm generate:api-client` - Generate API client from OpenAPI spec
- `pnpm generate:api` - Fetch OpenAPI spec and generate API client

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## 🔄 Data Fetching Strategy

> [!NOTE]
> You don't need to run the OpenAPI commands below to run the Front-end. You will only need to run them when adding or modifying endpoints on the Backend API and wanting to use those on the Frontend.

This project uses an auto-generated API client powered by [**Orval**](https://orval.dev/), which creates type-safe API clients from OpenAPI specifications.

### How It Works

1. **Backend Requirements**: Each API endpoint needs a summary and tag in the OpenAPI spec
2. **Operation ID Generation**: FastAPI generates operation IDs using the pattern `{method}{tag}{summary}`
3. **Spec Fetching**: The OpenAPI spec is fetched from `http://localhost:8006/openapi.json` and saved to the frontend
4. **Spec Transformation**: The OpenAPI spec is cleaned up using a custom transformer (see `autogpt_platform/frontend/src/app/api/transformers`)
5. **Client Generation**: Auto-generated client includes TypeScript types, API endpoints, and Zod schemas, organized by tags

### API Client Commands

```bash
# Fetch OpenAPI spec from backend and generate client
pnpm generate:api

# Only fetch the OpenAPI spec
pnpm fetch:openapi

# Only generate the client (after spec is fetched)
pnpm generate:api-client
```

### Using the Generated Client

The generated client provides React Query hooks for both queries and mutations:

#### Queries (GET requests)

```typescript
import { useGetV1GetNotificationPreferences } from "@/app/api/__generated__/endpoints/auth/auth";

const { data, isLoading, isError } = useGetV1GetNotificationPreferences({
  query: {
    select: (res) => res.data,
    // Other React Query options
  },
});
```

#### Mutations (POST, PUT, DELETE requests)

```typescript
import { useDeleteV2DeleteStoreSubmission } from "@/app/api/__generated__/endpoints/store/store";
import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import { useQueryClient } from "@tanstack/react-query";

const queryClient = useQueryClient();

const { mutateAsync: deleteSubmission } = useDeleteV2DeleteStoreSubmission({
  mutation: {
    onSuccess: () => {
      // Invalidate related queries to refresh data
      queryClient.invalidateQueries({
        queryKey: getGetV2ListMySubmissionsQueryKey(),
      });
    },
  },
});

// Usage
await deleteSubmission({
  submissionId: submission_id,
});
```

#### Server Actions

For server-side operations, you can also use the generated client functions directly:

```typescript
import { postV1UpdateNotificationPreferences } from "@/app/api/__generated__/endpoints/auth/auth";

// In a server action
const preferences = {
  email: "user@example.com",
  preferences: {
    AGENT_RUN: true,
    ZERO_BALANCE: false,
    // ... other preferences
  },
  daily_limit: 0,
};

await postV1UpdateNotificationPreferences(preferences);
```

#### Server-Side Prefetching

For server-side components, you can prefetch data on the server and hydrate it in the client cache. This allows immediate access to cached data when queries are called:

```typescript
import { getQueryClient } from "@/lib/tanstack-query/getQueryClient";
import {
  prefetchGetV2ListStoreAgentsQuery,
  prefetchGetV2ListStoreCreatorsQuery
} from "@/app/api/__generated__/endpoints/store/store";
import { HydrationBoundary, dehydrate } from "@tanstack/react-query";

// In your server component
const queryClient = getQueryClient();

await Promise.all([
  prefetchGetV2ListStoreAgentsQuery(queryClient, {
    featured: true,
  }),
  prefetchGetV2ListStoreAgentsQuery(queryClient, {
    sorted_by: "runs",
  }),
  prefetchGetV2ListStoreCreatorsQuery(queryClient, {
    featured: true,
    sorted_by: "num_agents",
  }),
]);

return (
  <HydrationBoundary state={dehydrate(queryClient)}>
    <MainMarkeplacePage />
  </HydrationBoundary>
);
```

This pattern improves performance by serving pre-fetched data from the server while maintaining the benefits of client-side React Query features.

### Configuration

The Orval configuration is located in `autogpt_platform/frontend/orval.config.ts`. It generates two separate clients:

1. **autogpt_api_client**: React Query hooks for client-side data fetching
2. **autogpt_zod_schema**: Zod schemas for validation

For more details, see the [Orval documentation](https://orval.dev/) or check the configuration file.

## 🚩 Feature Flags

This project uses [LaunchDarkly](https://launchdarkly.com/) for feature flags, allowing us to control feature rollouts and A/B testing.

### Using Feature Flags

#### Check if a feature is enabled

```typescript
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

function MyComponent() {
  const isAgentActivityEnabled = useGetFlag(Flag.AGENT_ACTIVITY);

  if (!isAgentActivityEnabled) {
    return null; // Hide feature
  }

  return <div>Feature is enabled!</div>;
}
```

#### Protect entire components

```typescript
import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";

const MyFeaturePage = withFeatureFlag(MyPageComponent, "my-feature-flag");
```

### Testing with Feature Flags

For local development or running Playwright tests locally, use mocked feature flags by setting `NEXT_PUBLIC_PW_TEST=true` in your `.env` file. This bypasses LaunchDarkly and uses the mock values defined in the code.

### Adding New Flags

1. Add the flag to the `Flag` enum in `use-get-flag.ts`
2. Add the flag type to `FlagValues` type
3. Add mock value to `mockFlags` for testing
4. Configure the flag in LaunchDarkly dashboard

## 🚚 Deploy

TODO

## 📙 Storybook

Storybook is a powerful development environment for UI components. It allows you to build UI components in isolation, making it easier to develop, test, and document your components independently from your main application.

### Purpose in the Development Process

1. **Component Development**: Develop and test UI components in isolation.
2. **Visual Testing**: Easily spot visual regressions.
3. **Documentation**: Automatically document components and their props.
4. **Collaboration**: Share components with your team or stakeholders for feedback.

### How to Use Storybook

1. **Start Storybook**:
   Run the following command to start the Storybook development server:

   ```bash
   pnpm storybook
   ```

   This will start Storybook on port 6006. Open [http://localhost:6006](http://localhost:6006) in your browser to view your component library.

2. **Build Storybook**:
   To build a static version of Storybook for deployment, use:

   ```bash
   pnpm build-storybook
   ```

3. **Running Storybook Tests**:
   Storybook tests can be run using:

   ```bash
   pnpm test-storybook
   ```

4. **Writing Stories**:
   Create `.stories.tsx` files alongside your components to define different states and variations of your components.

By integrating Storybook into our development workflow, we can streamline UI development, improve component reusability, and maintain a consistent design system across the project.

## 🔭 Tech Stack

### Core Framework & Language

- [**Next.js**](https://nextjs.org/) - React framework with App Router
- [**React**](https://react.dev/) - UI library for building user interfaces
- [**TypeScript**](https://www.typescriptlang.org/) - Typed JavaScript for better developer experience

### Styling & UI Components

- [**Tailwind CSS**](https://tailwindcss.com/) - Utility-first CSS framework
- [**shadcn/ui**](https://ui.shadcn.com/) - Re-usable components built with Radix UI and Tailwind CSS
- [**Radix UI**](https://www.radix-ui.com/) - Headless UI components for accessibility
- [**Lucide React**](https://lucide.dev/guide/packages/lucide-react) - Beautiful & consistent icons
- [**Framer Motion**](https://motion.dev/) - Animation library for React

### Development & Testing

- [**Storybook**](https://storybook.js.org/) - Component development environment
- [**Playwright**](https://playwright.dev/) - End-to-end testing framework
- [**ESLint**](https://eslint.org/) - JavaScript/TypeScript linting
- [**Prettier**](https://prettier.io/) - Code formatting

### Backend & Services

- [**Supabase**](https://supabase.com/) - Backend-as-a-Service (database, auth, storage)
- [**Sentry**](https://sentry.io/) - Error monitoring and performance tracking

### Package Management

- [**pnpm**](https://pnpm.io/) - Fast, disk space efficient package manager
- [**Corepack**](https://github.com/nodejs/corepack) - Node.js package manager management

### Additional Libraries

- [**React Hook Form**](https://react-hook-form.com/) - Forms with easy validation
- [**Zod**](https://zod.dev/) - TypeScript-first schema validation
- [**React Table**](https://tanstack.com/table) - Headless table library
- [**React Flow**](https://reactflow.dev/) - Interactive node-based diagrams
- [**React Query**](https://tanstack.com/query/latest/docs/framework/react/overview) - Data fetching and caching
- [**React Query DevTools**](https://tanstack.com/query/latest/docs/framework/react/devtools) - Debugging tool for React Query

### Development Tools

- `NEXT_PUBLIC_REACT_QUERY_DEVTOOL` - Enable React Query DevTools. Set to `true` to enable.
