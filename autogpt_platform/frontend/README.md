This is the frontend for AutoGPT's next generation

## Getting Started

Run the following installation once.

```bash
npm install
# or
yarn install
# or
pnpm install
# or
bun install
```

Next, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

For subsequent runs, you do not have to `npm install` again. Simply do `npm run dev`.

If the project is updated via git, you will need to `npm install` after each update.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Deploy

TODO

## Storybook

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
   npm run storybook
   ```

   This will start Storybook on port 6006. Open [http://localhost:6006](http://localhost:6006) in your browser to view your component library.

2. **Build Storybook**:
   To build a static version of Storybook for deployment, use:

   ```bash
   npm run build-storybook
   ```

3. **Running Storybook Tests**:
   Storybook tests can be run using:

   ```bash
   npm run test-storybook
   ```

   For CI environments, use:

   ```bash
   npm run test-storybook:ci
   ```

4. **Writing Stories**:
   Create `.stories.tsx` files alongside your components to define different states and variations of your components.

By integrating Storybook into our development workflow, we can streamline UI development, improve component reusability, and maintain a consistent design system across the project.
