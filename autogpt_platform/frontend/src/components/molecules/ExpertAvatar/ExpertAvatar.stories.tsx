import type { Meta, StoryObj } from "@storybook/nextjs";
import { ExpertAvatar } from "./ExpertAvatar";
import { BUILTIN_EXPERT_AVATARS } from "./helpers";
import { getExpertTopicHex } from "./colors";
import { ExpertCover } from "@/app/(platform)/team/components/ExpertTeamCard/components/ExpertCover";

const meta = {
  title: "Molecules/ExpertAvatar",
  component: ExpertAvatar,
  args: { name: "Expert", avatarUrl: null },
} satisfies Meta<typeof ExpertAvatar>;
export default meta;
type Story = StoryObj<typeof meta>;

const EXAMPLES = [
  ["jules", "Marketing"],
  ["remy", "Marketing"],
  ["quinn", "Research"],
  ["harper", "Operations"],
  ["vera", "Operations"],
  ["anika", "Sales"],
  ["marco", "Marketing"],
  ["lena", "Operations"],
  ["mina", "Finance"],
];

export const TopicSurfaces: Story = {
  render: function TopicSurfaces() {
    return (
      <div className="grid w-full max-w-4xl grid-cols-3 gap-4">
        {EXAMPLES.map(([id, role]) => {
          const avatar = BUILTIN_EXPERT_AVATARS.find(
            (avatar) => avatar.id === id,
          )!;
          const color = getExpertTopicHex(role);
          return (
            <div
              key={id}
              className="rounded-2xl border border-zinc-200 bg-white p-2"
            >
              <ExpertCover color={color} />
              <div className="relative -mt-12 px-3">
                <ExpertAvatar
                  name={avatar.name}
                  avatarUrl={avatar.url}
                  backgroundColor={color}
                  size={88}
                  className="ring-4 ring-white"
                />
                <p className="mt-2 font-semibold">{avatar.name}</p>
                <p className="text-sm text-zinc-500">{role}</p>
                <div className="my-3 flex items-center gap-2 text-sm text-zinc-500">
                  <ExpertAvatar
                    name={avatar.name}
                    avatarUrl={avatar.url}
                    size={32}
                    className="rounded-full border border-[#e3e3e3]"
                  />
                  Sidebar
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  },
};
