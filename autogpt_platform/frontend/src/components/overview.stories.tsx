import type { Meta, StoryObj } from "@storybook/nextjs";
import Image from "next/image";

function RightArrow() {
  return (
    <svg
      viewBox="0 0 14 14"
      width="8px"
      height="14px"
      className="ml-1 inline-block fill-current"
    >
      <path d="m11.1 7.35-5.5 5.5a.5.5 0 0 1-.7-.7L10.04 7 4.9 1.85a.5.5 0 1 1 .7-.7l5.5 5.5c.2.2.2.5 0 .7Z" />
    </svg>
  );
}

function OverviewComponent() {
  const linkStyle = "font-bold text-info-foreground hover:text-info-foreground";
  return (
    <div className="mx-auto max-w-6xl space-y-24 p-8">
      {/* Header Section */}
      <div className="space-y-8">
        <div className="space-y-4">
          <h1 className="text-title font-bold text-foreground-base">
            AutoGPT Design System
          </h1>
          <p className="text-lead leading-relaxed text-foreground-secondary">
            Welcome to the AutoGPT Design System - a comprehensive collection of
            reusable components, design tokens, and guidelines that power the
            AutoGPT Platform. This system ensures consistency, accessibility,
            and efficiency across all our user interfaces.
          </p>
          <div className="inline-flex items-center">
            <strong className="text-lead">
              <a
                href="https://www.figma.com/design/nO9NFynNuicLtkiwvOxrbz/AutoGPT-Design-System?node-id=3-2083&m=dev"
                target="_blank"
                rel="noopener noreferrer"
                className={linkStyle}
              >
                📋 Figma Reference
              </a>
            </strong>
          </div>
        </div>

        {/* Foundation Cards */}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <div className="flex flex-col space-y-4">
            <Image
              src="/storybook/tokens.png"
              alt="Design tokens representing colors, typography, and spacing"
              width={0}
              height={0}
              className="h-auto w-4/5 rounded-lg"
            />
            <h4 className="text-lead font-bold text-foreground-base">
              Design Tokens
            </h4>
            <p className="text-body leading-relaxed text-foreground-secondary">
              The foundation of our design system. Tokens define colors,
              typography, spacing, shadows, and other visual properties that
              ensure consistency across all components and layouts.
            </p>
            <a
              href="?path=/docs/design-tokens--docs"
              className={`inline-flex items-center text-body ${linkStyle}`}
            >
              Explore Tokens
              <RightArrow />
            </a>
          </div>

          <div className="flex flex-col space-y-4">
            <Image
              src="/storybook/atoms.png"
              alt="Basic UI elements like buttons, inputs, and icons"
              width={0}
              height={0}
              className="h-auto w-4/5 rounded-lg"
            />
            <h4 className="text-lead font-bold text-foreground-base">Atoms</h4>
            <p className="text-body leading-relaxed text-foreground-secondary">
              The smallest building blocks of our interface. Atoms include
              buttons, inputs, icons, labels, and other fundamental UI elements
              that cannot be broken down further.
            </p>
            <a
              href="?path=/docs/atoms--docs"
              className={`inline-flex items-center text-body ${linkStyle}`}
            >
              View Atoms
              <RightArrow />
            </a>
          </div>

          <div className="flex flex-col space-y-4">
            <Image
              src="/storybook/molecules.png"
              alt="Combined UI components like cards, dropdowns, and search bars"
              width={0}
              height={0}
              className="h-auto w-4/5 rounded-lg"
            />
            <h4 className="text-lead font-bold text-foreground-base">
              Molecules
            </h4>
            <p className="text-body leading-relaxed text-foreground-secondary">
              Combinations of atoms that work together as a unit. Examples
              include search bars, card components, dropdown menus, and other
              composite UI elements.
            </p>
            <a
              href="?path=/docs/molecules--docs"
              className={`inline-flex items-center text-body ${linkStyle}`}
            >
              Browse Molecules
              <RightArrow />
            </a>
          </div>
        </div>
      </div>

      {/* Technical Foundation */}
      <div className="space-y-8">
        <div className="space-y-4">
          <h2 className="text-heading font-bold text-foreground-base">
            Technical Foundation
          </h2>
          <p className="text-lead text-foreground-secondary">
            Our design system is built on proven technologies while maintaining
            strict design consistency through custom tokens and components.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-12 md:grid-cols-3">
          <div className="space-y-4">
            <h4 className="text-lead font-semibold text-foreground-base">
              🎨 Built with Tailwind & shadcn/ui
            </h4>
            <p className="text-body leading-relaxed text-foreground-secondary">
              The AutoGPT Design System leverages{" "}
              <a
                href="https://tailwindcss.com/"
                target="_blank"
                rel="noopener noreferrer"
                className={linkStyle}
              >
                Tailwind CSS
              </a>{" "}
              for utility-first styling and{" "}
              <a
                href="https://ui.shadcn.com/"
                target="_blank"
                rel="noopener noreferrer"
                className={linkStyle}
              >
                shadcn/ui
              </a>{" "}
              as a foundation for accessible, well-tested components.
            </p>
          </div>

          <div className="space-y-4">
            <h4 className="text-lead font-semibold text-foreground-base">
              🔧 Why This Matters
            </h4>
            <ul className="space-y-2 text-body text-foreground-secondary">
              <li>
                <strong>Visual Consistency:</strong> All interfaces look and
                feel cohesive
              </li>
              <li>
                <strong>Accessibility:</strong> Our tokens include proper
                contrast ratios and focus states
              </li>
              <li>
                <strong>Brand Compliance:</strong> All colors and styles match
                AutoGPT&apos;s brand
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <h4 className="text-lead font-semibold text-foreground-base">
              📚 Getting Started
            </h4>
            <ol className="list-decimal space-y-2 text-body text-foreground-secondary">
              <li>
                Review the Design Tokens, Atoms,Molecules and Contextual
                Components for your use case
              </li>
              <li>
                If you need something new,{" "}
                <a
                  href="https://github.com/Significant-Gravitas/AutoGPT/issues/new?template=design-system.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={linkStyle}
                >
                  create an issue first
                </a>{" "}
                and get feedback
              </li>
              <li>
                Always test your changes and ensure it renders well on all
                screen sizes
              </li>
            </ol>
          </div>
        </div>
      </div>

      {/* Contributing Section */}
      <div className="space-y-8">
        <div className="space-y-4">
          <h2 className="text-heading font-bold text-foreground-base">
            Contributing to the Design System
          </h2>
          <p className="text-lead text-foreground-secondary">
            Help us improve and expand the AutoGPT Design System. Whether
            you&apos;re fixing bugs, adding new components, or enhancing
            existing ones, your contributions are valuable to the community.
          </p>
        </div>

        <div className="relative rounded-xl bg-gradient-to-r from-info-surface via-accent-surface to-pink/10 p-6">
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-info via-brand to-pink p-[2px]">
            <div className="h-full w-full rounded-xl bg-surface"></div>
          </div>
          <div className="relative space-y-6">
            <div className="text-center">
              <h4 className="bg-gradient-to-r from-info-foreground via-brand-hover to-pink-foreground bg-clip-text text-lead font-bold text-transparent">
                ⚠️ Design System Guidelines
              </h4>
              <p className="mt-2 text-foreground-body">
                Contributors must <strong>ONLY</strong> use the design tokens
                and components defined in this system
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <h5 className="text-lead font-semibold text-error-foreground">
                  ❌ Don&apos;t Do This
                </h5>
                <div className="space-y-3">
                  <div>
                    <p className="mb-2 text-body font-medium text-foreground-body">
                      Default Tailwind classes:
                    </p>
                    <pre className="overflow-x-auto rounded-md bg-surface-hover p-3 text-body">
                      <code className="text-error-foreground">{`className="text-blue-500 p-4 bg-gray-200"`}</code>
                    </pre>
                  </div>
                  <div>
                    <p className="mb-2 text-body font-medium text-foreground-body">
                      Arbitrary values:
                    </p>
                    <pre className="overflow-x-auto rounded-md bg-surface-hover p-3 text-body">
                      <code className="text-error-foreground">{`className="text-[#1234ff] w-[420px]"`}</code>
                    </pre>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h5 className="text-lead font-semibold text-success">
                  ✅ Do This Instead
                </h5>
                <div className="space-y-3">
                  <div>
                    <p className="mb-2 text-body font-medium text-foreground-body">
                      Design tokens:
                    </p>
                    <pre className="overflow-x-auto rounded-md bg-surface-hover p-3 text-body">
                      <code className="text-success">{`className="text-action bg-surface space-4"`}</code>
                    </pre>
                  </div>
                  <div>
                    <p className="mb-2 text-body font-medium text-foreground-body">
                      System components:
                    </p>
                    <pre className="overflow-x-auto rounded-md bg-surface-hover p-3 text-body">
                      <code className="text-success">{`<Button variant="primary" size="md">
  Click me
</Button>`}</code>
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="space-y-4">
            <h4 className="text-lead font-semibold text-foreground-base">
              🧢 Development Workflow
            </h4>
            <p className="leading-relaxed text-foreground-secondary">
              All design system changes should follow our established workflow
              to ensure quality and consistency.
            </p>
            <div className="space-y-4">
              <div>
                <strong className="text-foreground-base">
                  For External Contributors:
                </strong>
                <ol className="mt-2 list-decimal space-y-1 pl-4 text-foreground-secondary">
                  <li>
                    Create a GitHub issue first to discuss your proposed changes
                  </li>
                  <li>
                    Wait for maintainer approval before starting development
                  </li>
                  <li>Fork the repository and create a feature branch</li>
                  <li>Implement changes following our coding standards</li>
                  <li>Submit a pull request with detailed description</li>
                </ol>
              </div>
              <div>
                <strong className="text-foreground-base">
                  For Internal Team Members:
                </strong>
                <ol className="mt-2 list-decimal space-y-1 pl-4 text-foreground-secondary">
                  <li>Create a feature branch from main</li>
                  <li>Implement changes and update Storybook documentation</li>
                  <li>Test components across different scenarios</li>
                  <li>Submit pull request for team review</li>
                </ol>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="text-lead font-semibold text-foreground-base">
              📋 Component Guidelines
            </h4>
            <p className="leading-relaxed text-foreground-secondary">
              Follow these principles when creating or modifying components:
            </p>
            <ul className="space-y-2 text-foreground-secondary">
              <li>
                <strong className="text-foreground-base">
                  Accessibility First
                </strong>
                <ul>
                  <li>
                    All components must meet{" "}
                    <a
                      href="https://www.w3.org/WAI/standards-guidelines/wcag/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`${linkStyle} font-semibold`}
                    >
                      WCAG 2.1 AA standards
                    </a>
                  </li>
                </ul>
              </li>
              <li>
                <strong className="text-foreground-base">
                  Design Token Usage
                </strong>
                <ul>
                  <li>Use design tokens for all styling properties</li>
                </ul>
              </li>
              <li>
                <strong className="text-foreground-base">
                  Responsive Design
                </strong>
                <ul>
                  <li>Components should work across all screen sizes</li>
                </ul>
              </li>
              <li>
                <strong className="text-foreground-base">TypeScript</strong>
                <ul>
                  <li>All components must be fully typed</li>
                </ul>
              </li>
              <li>
                <strong className="text-foreground-base">Documentation</strong>
                <ul>
                  <li>
                    Include comprehensive Storybook stories and JSDoc comments
                  </li>
                </ul>
              </li>
              <li>
                <strong className="text-foreground-base">Testing</strong>
                <ul>
                  <li>Write unit tests for component logic and interactions</li>
                </ul>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Social Links */}
      <div className="space-y-8">
        <h3 className="text-heading font-bold text-foreground-base">
          Get Involved
        </h3>
        <p className="leading-relaxed text-foreground-secondary">
          Join the AutoGPT community and help build the future of AI automation.
        </p>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-4">
          {[
            {
              icon: "/storybook/github.svg",
              title:
                "Contribute to the AutoGPT Platform and help build the future of AI automation.",
              link: "https://github.com/Significant-Gravitas/AutoGPT",
              linkText: "Star on GitHub",
            },
            {
              icon: "/storybook/discord.svg",
              title: "Get support and chat with the AutoGPT community.",
              link: "https://discord.gg/autogpt",
              linkText: "Join Discord",
            },
            {
              icon: "/storybook/youtube.svg",
              title: "Watch AutoGPT tutorials and feature demonstrations.",
              link: "https://www.youtube.com/@AutoGPT-Official",
              linkText: "Watch on YouTube",
            },
            {
              icon: "/storybook/docs.svg",
              title: "Read the complete platform documentation and guides.",
              link: "https://docs.agpt.co",
              linkText: "View Documentation",
            },
          ].map((item, index) => (
            <div key={index} className="space-y-4">
              <Image
                src={item.icon}
                alt={`${item.linkText} logo`}
                width={32}
                height={32}
                className="h-8 w-8"
              />
              <p className="text-body leading-relaxed text-foreground-secondary">
                {item.title}
              </p>
              <a
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-body text-info-foreground hover:text-info-foreground"
              >
                {item.linkText}
                <RightArrow />
              </a>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const meta: Meta<typeof OverviewComponent> = {
  title: "Overview",
  component: OverviewComponent,
  parameters: {
    layout: "fullscreen",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
