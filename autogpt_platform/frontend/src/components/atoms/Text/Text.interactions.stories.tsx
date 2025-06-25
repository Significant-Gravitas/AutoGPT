import type { Meta, StoryObj } from "@storybook/nextjs";
import { expect, userEvent, within } from "storybook/test";
import { Text, textVariants } from "./Text";

const meta: Meta<typeof Text> = {
  title: "Atoms/Tests/Text",
  component: Text,
  parameters: {
    layout: "centered",
  },
  tags: ["!autodocs"],
};

export default meta;
type Story = StoryObj<typeof meta>;

// Test basic text rendering
export const BasicTextInteraction: Story = {
  args: {
    variant: "body",
    children: "Basic text content",
  },
  play: async function testBasicText({ canvasElement }) {
    const canvas = within(canvasElement);
    const text = canvas.getByText("Basic text content");

    // Test basic rendering
    expect(text).toBeInTheDocument();
    expect(text).toBeVisible();
    expect(text).toHaveTextContent("Basic text content");
  },
};

// Test all text variants render correctly
export const AllVariantsInteraction: Story = {
  render: function renderAllVariants() {
    return (
      <div className="space-y-4">
        {textVariants.map((variant) => (
          <Text key={variant} variant={variant}>
            {variant}: Sample text for {variant} variant
          </Text>
        ))}
      </div>
    );
  },
  play: async function testAllVariants({ canvasElement }) {
    const canvas = within(canvasElement);

    // Test all variants are rendered
    for (const variant of textVariants) {
      const text = canvas.getByText(
        new RegExp(`${variant}: Sample text for ${variant} variant`),
      );
      expect(text).toBeInTheDocument();
    }

    // Test specific heading elements
    const h1 = canvas.getByText(/h1: Sample text/);
    const h2 = canvas.getByText(/h2: Sample text/);
    const h3 = canvas.getByText(/h3: Sample text/);
    const h4 = canvas.getByText(/h4: Sample text/);

    expect(h1.tagName).toBe("H1");
    expect(h2.tagName).toBe("H2");
    expect(h3.tagName).toBe("H3");
    expect(h4.tagName).toBe("H4");
  },
};

// Test custom element override
export const CustomElementInteraction: Story = {
  args: {
    variant: "body",
    as: "span",
    children: "Text as span element",
  },
  play: async function testCustomElement({ canvasElement }) {
    const canvas = within(canvasElement);
    const text = canvas.getByText("Text as span element");

    // Test custom element type
    expect(text.tagName).toBe("SPAN");
    expect(text).toHaveTextContent("Text as span element");
  },
};

// Test text with custom classes
export const CustomClassesInteraction: Story = {
  args: {
    variant: "body",
    children: "Text with custom classes",
    className: "text-red-500 underline font-bold",
  },
  play: async function testCustomClasses({ canvasElement }) {
    const canvas = within(canvasElement);
    const text = canvas.getByText("Text with custom classes");

    // Test custom classes are applied
    expect(text).toHaveClass("text-red-500", "underline", "font-bold");

    // Test variant classes are still present
    expect(text).toHaveClass("font-sans", "text-sm");
  },
};

// Test text accessibility
export const AccessibilityInteraction: Story = {
  render: function renderAccessibilityTest() {
    return (
      <div>
        <Text variant="h1" role="heading" aria-level={1}>
          Main Heading
        </Text>
        <Text variant="body">Descriptive text content</Text>
        <Text variant="small" aria-label="Helper text">
          SR: Screen reader text
        </Text>
      </div>
    );
  },
  play: async function testAccessibility({ canvasElement }) {
    const canvas = within(canvasElement);

    // Test heading accessibility
    const heading = canvas.getByRole("heading", { level: 1 });
    expect(heading).toHaveTextContent("Main Heading");

    // Test aria-label
    const helperText = canvas.getByLabelText("Helper text");
    expect(helperText).toHaveTextContent("SR: Screen reader text");

    // Test regular text
    const bodyText = canvas.getByText("Descriptive text content");
    expect(bodyText).toBeInTheDocument();
  },
};

// Test text selection
export const TextSelectionInteraction: Story = {
  args: {
    variant: "large",
    children: "This text can be selected and copied",
  },
  play: async function testTextSelection({ canvasElement }) {
    const canvas = within(canvasElement);
    const text = canvas.getByText("This text can be selected and copied");

    // Test text selection (simulated)
    // Note: Actual text selection is limited in test environment
    await userEvent.click(text);

    // Test double-click for word selection
    await userEvent.dblClick(text);

    // Verify text is still accessible after interactions
    expect(text).toBeInTheDocument();
    expect(text).toHaveTextContent("This text can be selected and copied");
  },
};

// Test text with HTML content (should be escaped)
export const SafeHTMLInteraction: Story = {
  args: {
    variant: "body",
    children: "<script>alert('xss')</script>Safe text content",
  },
  play: async function testSafeHTML({ canvasElement }) {
    const canvas = within(canvasElement);
    const text = canvas.getByText(
      "<script>alert('xss')</script>Safe text content",
    );

    // Test that HTML is escaped/rendered as text
    expect(text).toHaveTextContent(
      "<script>alert('xss')</script>Safe text content",
    );

    // Test that no script elements are created
    const scripts = canvasElement.querySelectorAll("script");
    expect(scripts).toHaveLength(0);
  },
};
