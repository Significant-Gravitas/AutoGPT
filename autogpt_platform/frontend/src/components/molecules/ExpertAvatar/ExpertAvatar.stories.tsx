import type { Meta, StoryObj } from "@storybook/nextjs";
import { ExpertAvatar } from "./ExpertAvatar";
import { MANAGED_IDENTITIES } from "./helpers";
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
  "expert-maria",
  "expert-jules",
  "expert-remy",
  "expert-quinn",
  "expert-harper",
  "expert-frankie",
  "expert-anika",
  "expert-devon",
  "expert-mina",
];

/** One identity across the surfaces that show it: the team cover with the
 *  category wash, the 88 px card avatar and the 32 px sidebar avatar. The
 *  artwork is the same managed tile at every size. */
export const TopicSurfaces: Story = {
  render: function TopicSurfaces() {
    return (
      <div className="grid w-full max-w-4xl grid-cols-3 gap-4">
        {EXAMPLES.map((id) => {
          const identity = MANAGED_IDENTITIES.find(
            (identity) => identity.id === id,
          )!;
          const color = getExpertTopicHex({
            avatarUrl: identity.url,
            categories: [identity.visual_category],
          });
          return (
            <div
              key={id}
              className="rounded-2xl border border-zinc-200 bg-white p-2"
            >
              <ExpertCover color={color} />
              <div className="relative -mt-12 px-3">
                <ExpertAvatar
                  name={identity.name}
                  avatarUrl={identity.url}
                  backgroundColor={color}
                  size={88}
                  className="rounded-full ring-4 ring-white"
                />
                <p className="mt-2 font-semibold">{identity.name}</p>
                <p className="text-sm text-zinc-500">{identity.job_title}</p>
                <div className="my-3 flex items-center gap-2 text-sm text-zinc-500">
                  <ExpertAvatar
                    name={identity.name}
                    avatarUrl={identity.url}
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

/** Every managed identity at the sizes the product uses, grouped by palette
 *  family: category peers share one material color and differ in silhouette
 *  and cream route. */
export const Family: Story = {
  render: function Family() {
    const families = Array.from(
      new Set(MANAGED_IDENTITIES.map((identity) => identity.visual_category)),
    );
    return (
      <div className="flex w-full max-w-5xl flex-col gap-6">
        {families.map((family) => (
          <div key={family}>
            <p className="mb-2 text-sm font-medium capitalize text-zinc-500">
              {family}
            </p>
            <div className="flex flex-wrap gap-4">
              {MANAGED_IDENTITIES.filter(
                (identity) => identity.visual_category === family,
              ).map((identity) => (
                <div key={identity.id} className="flex items-end gap-2">
                  <ExpertAvatar
                    name={identity.name}
                    avatarUrl={identity.url}
                    size={96}
                    className="rounded-2xl"
                  />
                  <ExpertAvatar
                    name={identity.name}
                    avatarUrl={identity.url}
                    size={32}
                    className="rounded-full"
                  />
                  <ExpertAvatar
                    name={identity.name}
                    avatarUrl={identity.url}
                    size={24}
                    className="rounded-full"
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  },
};
