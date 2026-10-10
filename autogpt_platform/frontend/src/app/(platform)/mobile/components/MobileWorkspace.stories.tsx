import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { useNativePush } from "@/services/push-notifications/native/useNativePush";
import {
  attentionFixture,
  chatsFixture,
  expertFixture,
} from "../__tests__/fixtures";
import { MobileWorkspace } from "./MobileWorkspace";

let remainingAttention = attentionFixture;

const meta = {
  title: "Mobile/Workspace",
  component: MobileWorkspace,
  parameters: {
    layout: "fullscreen",
    msw: {
      handlers: [
        http.get("*/api/experts/identities", () =>
          HttpResponse.json([
            expertFixture,
            {
              ...expertFixture,
              id: "alex",
              name: "Alex",
              role: "Research analyst",
            },
          ]),
        ),
        http.get("*/api/chat/sessions", () => HttpResponse.json(chatsFixture)),
        http.get("*/api/home", () =>
          HttpResponse.json({ attention: remainingAttention }),
        ),
        http.post("*/api/review/action", async ({ request }) => {
          const body = (await request.json()) as {
            reviews: { node_exec_id: string }[];
          };
          remainingAttention = remainingAttention.filter(
            (item) =>
              !body.reviews.some(
                (review) => review.node_exec_id === item.review?.node_exec_id,
              ),
          );
          return HttpResponse.json({
            processed_count: 1,
            failed_count: 0,
            error: null,
          });
        }),
      ],
    },
  },
  loaders: [
    () => {
      remainingAttention = attentionFixture;
      useAuthStore.setState({
        user: {
          id: "fixture-user",
          email: "fixture@example.invalid",
          user_metadata: {},
        },
        hasLoadedUser: true,
        isUserLoading: false,
        initializationPromise: Promise.resolve(),
      });
      return {};
    },
  ],
  decorators: [
    (Story) => (
      <BackendAPIProvider>
        <TooltipProvider>
          <div className="-m-8 min-h-screen bg-zinc-50">
            <NativeFixturePush />
            <div className="px-4 py-2 text-center text-xs text-zinc-500">
              Local UI fixture · no live account
            </div>
            <Story />
          </div>
        </TooltipProvider>
      </BackendAPIProvider>
    ),
  ],
} satisfies Meta<typeof MobileWorkspace>;

export default meta;
type Story = StoryObj<typeof meta>;
function NativeFixturePush() {
  useNativePush();
  return null;
}
export const Chats: Story = { args: { tab: "chats" } };
export const Experts: Story = { args: { tab: "experts" } };
export const NeedsYou: Story = { args: { tab: "attention" } };
