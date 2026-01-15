This is the frontend for AutoGPT's next generation

## 🧢 Getting Started

This project uses [**pnpm**](https://pnpm.io/) as the package manager via **corepack**. [Corepack](https://github.com/nodejs/corepack) is a Node.js tool that automatically manages package managers without requiring global installations.

For architecture, conventions, data fetching, feature flags, design system usage, state management, and PR process, see [CONTRIBUTING.md](./CONTRIBUTING.md).
For Playwright and Storybook testing setup, see [TESTING.md](./TESTING.md).

### Prerequisites

Make sure you have Node.js 16.10+ installed. Corepack is included with Node.js by default.

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

## 🔄 Data Fetching

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidance on generated API hooks, SSR + hydration patterns, and usage examples. You generally do not need to run OpenAPI commands unless adding/modifying backend endpoints.

## 🚩 Feature Flags

See [CONTRIBUTING.md](./CONTRIBUTING.md) for feature flag usage patterns, local development with mocks, and how to add new flags.

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
- [**Phosphor Icons**](https://phosphoricons.com/) - Icon set used across the app
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
