import type { Meta } from "@storybook/nextjs";
import {
  Alert01Icon,
  ArrowDown01Icon,
  Loading03Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { StoryCode } from "@/components/tokens/helpers/StoryCode";
import { tones, type Tone } from "@/components/styles/tones";
import { Text } from "../Text/Text";
import { Icon } from "./Icon";

const meta: Meta<typeof Icon> = {
  title: "Atoms/Icon",
  component: Icon,
  tags: ["autodocs"],
  parameters: {
    docs: {
      description: {
        component:
          "Renders a Hugeicon at the design-system stroke width. `tone` sets its colour from the same roles as Text; without it the icon takes the colour around it. `spin` and `motion` animate it. className is for layout (size, margin, rotation).",
      },
    },
  },
};

export default meta;

export function Tones() {
  return (
    <div className="grid grid-cols-4 gap-4 p-8">
      {(Object.keys(tones) as Tone[]).map((tone) => (
        <div key={tone} className="flex items-center gap-2">
          <Icon icon={Tick02Icon} size={20} tone={tone} />
          <Text variant="small" tone="muted">
            {tone}
          </Text>
        </div>
      ))}
    </div>
  );
}

export function Motion() {
  return (
    <div className="flex items-center gap-8 p-8">
      <Icon icon={Loading03Icon} size={20} spin tone="muted" />
      <button type="button" className="group flex items-center gap-2">
        <Text variant="body" as="span">
          Hover to turn
        </Text>
        <Icon
          icon={ArrowDown01Icon}
          size={16}
          motion="transform"
          className="group-hover:rotate-180"
        />
      </button>
      <Icon icon={Alert01Icon} size={20} tone="warning-bright" />
      <StoryCode
        code={`<Icon icon={Loading03Icon} spin tone="muted" />
<Icon icon={ArrowDown01Icon} motion="transform" className={open ? "rotate-180" : undefined} />`}
      />
    </div>
  );
}
