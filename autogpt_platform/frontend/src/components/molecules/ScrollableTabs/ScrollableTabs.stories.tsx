import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  ScrollableTabs,
  ScrollableTabsContent,
  ScrollableTabsList,
  ScrollableTabsTrigger,
} from "./ScrollableTabs";

const meta = {
  title: "Molecules/ScrollableTabs",
  component: ScrollableTabs,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
  argTypes: {},
} satisfies Meta<typeof ScrollableTabs>;

export default meta;
type Story = StoryObj<typeof meta>;

function ScrollableTabsDemo() {
  return (
    <div className="flex flex-col gap-8 p-8">
      <h2 className="text-2xl font-bold">ScrollableTabs Examples</h2>

      <div className="space-y-6">
        <div>
          <h3 className="mb-4 text-lg font-semibold">
            Short Content (Tabs Hidden)
          </h3>
          <div className="h-[300px] overflow-y-auto border border-zinc-200">
            <ScrollableTabs defaultValue="tab1" className="h-full">
              <ScrollableTabsList>
                <ScrollableTabsTrigger value="tab1">
                  Account
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="tab2">
                  Password
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="tab3">
                  Settings
                </ScrollableTabsTrigger>
              </ScrollableTabsList>
              <ScrollableTabsContent value="tab1">
                <div className="p-4 text-sm">
                  Make changes to your account here. Click save when you&apos;re
                  done.
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="tab2">
                <div className="p-4 text-sm">
                  Change your password here. After saving, you&apos;ll be logged
                  out.
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="tab3">
                <div className="p-4 text-sm">
                  Update your preferences and settings here.
                </div>
              </ScrollableTabsContent>
            </ScrollableTabs>
          </div>
        </div>

        <div>
          <h3 className="mb-4 text-lg font-semibold">
            Long Content (Tabs Visible)
          </h3>
          <div className="h-[400px] overflow-y-auto border border-zinc-200">
            <ScrollableTabs defaultValue="tab1" className="h-full">
              <ScrollableTabsList>
                <ScrollableTabsTrigger value="tab1">
                  Account
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="tab2">
                  Password
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="tab3">
                  Settings
                </ScrollableTabsTrigger>
              </ScrollableTabsList>
              <ScrollableTabsContent value="tab1">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">
                    Account Settings
                  </h4>
                  <p className="mb-4">
                    Make changes to your account here. Click save when
                    you&apos;re done.
                  </p>
                  <p className="mb-4">
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed
                    do eiusmod tempor incididunt ut labore et dolore magna
                    aliqua. Ut enim ad minim veniam, quis nostrud exercitation
                    ullamco laboris.
                  </p>
                  <p className="mb-4">
                    Duis aute irure dolor in reprehenderit in voluptate velit
                    esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
                    occaecat cupidatat non proident.
                  </p>
                  <p>
                    Sed ut perspiciatis unde omnis iste natus error sit
                    voluptatem accusantium doloremque laudantium, totam rem
                    aperiam.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="tab2">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">
                    Password Settings
                  </h4>
                  <p className="mb-4">
                    Change your password here. After saving, you&apos;ll be
                    logged out.
                  </p>
                  <p className="mb-4">
                    At vero eos et accusamus et iusto odio dignissimos ducimus
                    qui blanditiis praesentium voluptatum deleniti atque
                    corrupti quos dolores et quas molestias excepturi sint
                    occaecati cupiditate.
                  </p>
                  <p className="mb-4">
                    Et harum quidem rerum facilis est et expedita distinctio.
                    Nam libero tempore, cum soluta nobis est eligendi optio
                    cumque nihil impedit quo minus.
                  </p>
                  <p>
                    Temporibus autem quibusdam et aut officiis debitis aut rerum
                    necessitatibus saepe eveniet ut et voluptates repudiandae
                    sint.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="tab3">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">
                    General Settings
                  </h4>
                  <p className="mb-4">
                    Update your preferences and settings here.
                  </p>
                  <p className="mb-4">
                    Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut
                    odit aut fugit, sed quia consequuntur magni dolores eos qui
                    ratione voluptatem sequi nesciunt.
                  </p>
                  <p className="mb-4">
                    Neque porro quisquam est, qui dolorem ipsum quia dolor sit
                    amet, consectetur, adipisci velit, sed quia non numquam eius
                    modi tempora incidunt ut labore et dolore magnam aliquam
                    quaerat voluptatem.
                  </p>
                  <p>
                    Ut enim ad minima veniam, quis nostrum exercitationem ullam
                    corporis suscipit laboriosam, nisi ut aliquid ex ea commodi
                    consequatur.
                  </p>
                </div>
              </ScrollableTabsContent>
            </ScrollableTabs>
          </div>
        </div>

        <div>
          <h3 className="mb-4 text-lg font-semibold">Many Tabs</h3>
          <div className="h-[500px] overflow-y-auto border border-zinc-200">
            <ScrollableTabs defaultValue="overview" className="h-full">
              <ScrollableTabsList>
                <ScrollableTabsTrigger value="overview">
                  Overview
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="analytics">
                  Analytics
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="reports">
                  Reports
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="notifications">
                  Notifications
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="integrations">
                  Integrations
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="billing">
                  Billing
                </ScrollableTabsTrigger>
              </ScrollableTabsList>
              <ScrollableTabsContent value="overview">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">
                    Dashboard Overview
                  </h4>
                  <p className="mb-4">
                    Dashboard overview with key metrics and recent activity.
                  </p>
                  <p className="mb-4">
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed
                    do eiusmod tempor incididunt ut labore et dolore magna
                    aliqua.
                  </p>
                  <p>
                    Ut enim ad minim veniam, quis nostrud exercitation ullamco
                    laboris nisi ut aliquip ex ea commodo consequat.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="analytics">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">Analytics</h4>
                  <p className="mb-4">
                    Detailed analytics and performance metrics.
                  </p>
                  <p className="mb-4">
                    Duis aute irure dolor in reprehenderit in voluptate velit
                    esse cillum dolore eu fugiat nulla pariatur.
                  </p>
                  <p>
                    Excepteur sint occaecat cupidatat non proident, sunt in
                    culpa qui officia deserunt mollit anim id est laborum.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="reports">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">Reports</h4>
                  <p className="mb-4">
                    Generate and view reports for your account.
                  </p>
                  <p className="mb-4">
                    Sed ut perspiciatis unde omnis iste natus error sit
                    voluptatem accusantium doloremque laudantium.
                  </p>
                  <p>
                    Totam rem aperiam, eaque ipsa quae ab illo inventore
                    veritatis et quasi architecto beatae vitae dicta sunt
                    explicabo.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="notifications">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">Notifications</h4>
                  <p className="mb-4">Manage your notification preferences.</p>
                  <p className="mb-4">
                    Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut
                    odit aut fugit.
                  </p>
                  <p>
                    Sed quia consequuntur magni dolores eos qui ratione
                    voluptatem sequi nesciunt.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="integrations">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">Integrations</h4>
                  <p className="mb-4">
                    Connect and manage third-party integrations.
                  </p>
                  <p className="mb-4">
                    Neque porro quisquam est, qui dolorem ipsum quia dolor sit
                    amet.
                  </p>
                  <p>
                    Consectetur, adipisci velit, sed quia non numquam eius modi
                    tempora incidunt.
                  </p>
                </div>
              </ScrollableTabsContent>
              <ScrollableTabsContent value="billing">
                <div className="p-8 text-sm">
                  <h4 className="mb-4 text-lg font-semibold">Billing</h4>
                  <p className="mb-4">
                    View and manage your billing information.
                  </p>
                  <p className="mb-4">
                    Ut enim ad minima veniam, quis nostrum exercitationem ullam
                    corporis suscipit laboriosam.
                  </p>
                  <p>
                    Nisi ut aliquid ex ea commodi consequatur? Quis autem vel
                    eum iure reprehenderit qui in ea voluptate velit esse.
                  </p>
                </div>
              </ScrollableTabsContent>
            </ScrollableTabs>
          </div>
        </div>
      </div>
    </div>
  );
}

export const Default = {
  render: () => <ScrollableTabsDemo />,
} satisfies Story;

export const ShortContent = {
  render: () => (
    <div className="p-8">
      <div className="h-[200px] overflow-y-auto border border-zinc-200">
        <ScrollableTabs defaultValue="account" className="h-full">
          <ScrollableTabsList>
            <ScrollableTabsTrigger value="account">
              Account
            </ScrollableTabsTrigger>
            <ScrollableTabsTrigger value="password">
              Password
            </ScrollableTabsTrigger>
          </ScrollableTabsList>
          <ScrollableTabsContent value="account">
            <div className="p-4 text-sm">
              Make changes to your account here. Click save when you&apos;re
              done.
            </div>
          </ScrollableTabsContent>
          <ScrollableTabsContent value="password">
            <div className="p-4 text-sm">
              Change your password here. After saving, you&apos;ll be logged
              out.
            </div>
          </ScrollableTabsContent>
        </ScrollableTabs>
      </div>
    </div>
  ),
} satisfies Story;

export const LongContent = {
  render: () => (
    <div className="p-8">
      <div className="h-[600px] overflow-y-auto border border-zinc-200">
        <ScrollableTabs defaultValue="tab1" className="h-full">
          <ScrollableTabsList>
            <ScrollableTabsTrigger value="tab1">Account</ScrollableTabsTrigger>
            <ScrollableTabsTrigger value="tab2">Password</ScrollableTabsTrigger>
            <ScrollableTabsTrigger value="tab3">Settings</ScrollableTabsTrigger>
          </ScrollableTabsList>
          <ScrollableTabsContent value="tab1">
            <div className="p-8 text-sm">
              <h4 className="mb-4 text-lg font-semibold">Account Settings</h4>
              <p className="mb-4">
                Make changes to your account here. Click save when you&apos;re
                done.
              </p>
              <p className="mb-4">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
                eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
                enim ad minim veniam, quis nostrud exercitation ullamco laboris
                nisi ut aliquip ex ea commodo consequat.
              </p>
              <p className="mb-4">
                Duis aute irure dolor in reprehenderit in voluptate velit esse
                cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
                cupidatat non proident, sunt in culpa qui officia deserunt
                mollit anim id est laborum.
              </p>
              <p className="mb-4">
                Sed ut perspiciatis unde omnis iste natus error sit voluptatem
                accusantium doloremque laudantium, totam rem aperiam, eaque ipsa
                quae ab illo inventore veritatis et quasi architecto beatae
                vitae dicta sunt explicabo.
              </p>
              <p>
                Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit
                aut fugit, sed quia consequuntur magni dolores eos qui ratione
                voluptatem sequi nesciunt.
              </p>
            </div>
          </ScrollableTabsContent>
          <ScrollableTabsContent value="tab2">
            <div className="p-8 text-sm">
              <h4 className="mb-4 text-lg font-semibold">Password Settings</h4>
              <p className="mb-4">
                Change your password here. After saving, you&apos;ll be logged
                out.
              </p>
              <p className="mb-4">
                At vero eos et accusamus et iusto odio dignissimos ducimus qui
                blanditiis praesentium voluptatum deleniti atque corrupti quos
                dolores et quas molestias excepturi sint occaecati cupiditate
                non provident.
              </p>
              <p className="mb-4">
                Similique sunt in culpa qui officia deserunt mollitia animi, id
                est laborum et dolorum fuga. Et harum quidem rerum facilis est
                et expedita distinctio.
              </p>
              <p className="mb-4">
                Nam libero tempore, cum soluta nobis est eligendi optio cumque
                nihil impedit quo minus id quod maxime placeat facere possimus,
                omnis voluptas assumenda est, omnis dolor repellendus.
              </p>
              <p>
                Temporibus autem quibusdam et aut officiis debitis aut rerum
                necessitatibus saepe eveniet ut et voluptates repudiandae sint
                et molestiae non recusandae.
              </p>
            </div>
          </ScrollableTabsContent>
          <ScrollableTabsContent value="tab3">
            <div className="p-8 text-sm">
              <h4 className="mb-4 text-lg font-semibold">General Settings</h4>
              <p className="mb-4">Update your preferences and settings here.</p>
              <p className="mb-4">
                Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet,
                consectetur, adipisci velit, sed quia non numquam eius modi
                tempora incidunt ut labore et dolore magnam aliquam quaerat
                voluptatem.
              </p>
              <p className="mb-4">
                Ut enim ad minima veniam, quis nostrum exercitationem ullam
                corporis suscipit laboriosam, nisi ut aliquid ex ea commodi
                consequatur? Quis autem vel eum iure reprehenderit qui in ea
                voluptate velit esse quam nihil molestiae consequatur.
              </p>
              <p className="mb-4">
                Vel illum qui dolorem eum fugiat quo voluptas nulla pariatur? At
                vero eos et accusamus et iusto odio dignissimos ducimus qui
                blanditiis praesentium voluptatum deleniti atque corrupti quos
                dolores.
              </p>
              <p>
                Et quas molestias excepturi sint occaecati cupiditate non
                provident, similique sunt in culpa qui officia deserunt mollitia
                animi, id est laborum et dolorum fuga.
              </p>
            </div>
          </ScrollableTabsContent>
        </ScrollableTabs>
      </div>
    </div>
  ),
} satisfies Story;
