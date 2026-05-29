import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "./TabsLine";

const meta = {
  title: "Molecules/TabsLine",
  component: TabsLine,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof TabsLine>;

export default meta;
type Story = StoryObj<typeof meta>;

// Helper component to demonstrate tabs functionality
function TabsDemo() {
  return (
    <div className="flex flex-col gap-8 p-8">
      <h2 className="text-2xl font-bold">TabsLine Examples</h2>

      <div className="space-y-6">
        <div>
          <h3 className="mb-4 text-lg font-semibold">Basic Tabs</h3>
          <TabsLine defaultValue="tab1" className="w-full">
            <TabsLineList>
              <TabsLineTrigger value="tab1">Account</TabsLineTrigger>
              <TabsLineTrigger value="tab2">Password</TabsLineTrigger>
              <TabsLineTrigger value="tab3">Settings</TabsLineTrigger>
            </TabsLineList>
            <TabsLineContent value="tab1">
              <div className="p-4 text-sm">
                Make changes to your account here. Click save when you&apos;re
                done.
              </div>
            </TabsLineContent>
            <TabsLineContent value="tab2">
              <div className="p-4 text-sm">
                Change your password here. After saving, you&apos;ll be logged
                out.
              </div>
            </TabsLineContent>
            <TabsLineContent value="tab3">
              <div className="p-4 text-sm">
                Update your preferences and settings here.
              </div>
            </TabsLineContent>
          </TabsLine>
        </div>

        <div>
          <h3 className="mb-4 text-lg font-semibold">Many Tabs</h3>
          <TabsLine defaultValue="overview" className="w-full">
            <TabsLineList>
              <TabsLineTrigger value="overview">Overview</TabsLineTrigger>
              <TabsLineTrigger value="analytics">Analytics</TabsLineTrigger>
              <TabsLineTrigger value="reports">Reports</TabsLineTrigger>
              <TabsLineTrigger value="notifications">
                Notifications
              </TabsLineTrigger>
              <TabsLineTrigger value="integrations">
                Integrations
              </TabsLineTrigger>
              <TabsLineTrigger value="billing">Billing</TabsLineTrigger>
            </TabsLineList>
            <TabsLineContent value="overview">
              <div className="p-4 text-sm">
                Dashboard overview with key metrics and recent activity.
              </div>
            </TabsLineContent>
            <TabsLineContent value="analytics">
              <div className="p-4 text-sm">
                Detailed analytics and performance metrics.
              </div>
            </TabsLineContent>
            <TabsLineContent value="reports">
              <div className="p-4 text-sm">
                Generate and view reports for your account.
              </div>
            </TabsLineContent>
            <TabsLineContent value="notifications">
              <div className="p-4 text-sm">
                Manage your notification preferences.
              </div>
            </TabsLineContent>
            <TabsLineContent value="integrations">
              <div className="p-4 text-sm">
                Connect and manage third-party integrations.
              </div>
            </TabsLineContent>
            <TabsLineContent value="billing">
              <div className="p-4 text-sm">
                View and manage your billing information.
              </div>
            </TabsLineContent>
          </TabsLine>
        </div>

        <div>
          <h3 className="mb-4 text-lg font-semibold">Disabled Tab</h3>
          <TabsLine defaultValue="active1" className="w-full">
            <TabsLineList>
              <TabsLineTrigger value="active1">Active Tab</TabsLineTrigger>
              <TabsLineTrigger value="disabled" disabled>
                Disabled Tab
              </TabsLineTrigger>
              <TabsLineTrigger value="active2">Another Active</TabsLineTrigger>
            </TabsLineList>
            <TabsLineContent value="active1">
              <div className="p-4 text-sm">
                This is an active tab that you can interact with.
              </div>
            </TabsLineContent>
            <TabsLineContent value="disabled">
              <div className="p-4 text-sm">
                This content is for the disabled tab.
              </div>
            </TabsLineContent>
            <TabsLineContent value="active2">
              <div className="p-4 text-sm">
                Another active tab with different content.
              </div>
            </TabsLineContent>
          </TabsLine>
        </div>
      </div>
    </div>
  );
}

export const Default: Story = {
  render: () => <TabsDemo />,
};

export const BasicTabs: Story = {
  render: () => (
    <div className="p-8">
      <TabsLine defaultValue="account" className="w-[400px]">
        <TabsLineList>
          <TabsLineTrigger value="account">Account</TabsLineTrigger>
          <TabsLineTrigger value="password">Password</TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent value="account">
          <div className="p-4 text-sm">
            Make changes to your account here. Click save when you&apos;re done.
          </div>
        </TabsLineContent>
        <TabsLineContent value="password">
          <div className="p-4 text-sm">
            Change your password here. After saving, you&apos;ll be logged out.
          </div>
        </TabsLineContent>
      </TabsLine>
    </div>
  ),
};

export const ManyTabs: Story = {
  render: () => (
    <div className="p-8">
      <TabsLine defaultValue="home" className="w-full">
        <TabsLineList>
          <TabsLineTrigger value="home">Home</TabsLineTrigger>
          <TabsLineTrigger value="about">About</TabsLineTrigger>
          <TabsLineTrigger value="services">Services</TabsLineTrigger>
          <TabsLineTrigger value="portfolio">Portfolio</TabsLineTrigger>
          <TabsLineTrigger value="contact">Contact</TabsLineTrigger>
          <TabsLineTrigger value="blog">Blog</TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent value="home">
          <div className="p-4 text-sm">Welcome to our homepage!</div>
        </TabsLineContent>
        <TabsLineContent value="about">
          <div className="p-4 text-sm">Learn more about our company.</div>
        </TabsLineContent>
        <TabsLineContent value="services">
          <div className="p-4 text-sm">Discover our range of services.</div>
        </TabsLineContent>
        <TabsLineContent value="portfolio">
          <div className="p-4 text-sm">
            View our previous work and projects.
          </div>
        </TabsLineContent>
        <TabsLineContent value="contact">
          <div className="p-4 text-sm">Get in touch with us today.</div>
        </TabsLineContent>
        <TabsLineContent value="blog">
          <div className="p-4 text-sm">Read our latest blog posts.</div>
        </TabsLineContent>
      </TabsLine>
    </div>
  ),
};

export const WithDisabledTab: Story = {
  render: () => (
    <div className="p-8">
      <TabsLine defaultValue="available" className="w-[400px]">
        <TabsLineList>
          <TabsLineTrigger value="available">Available</TabsLineTrigger>
          <TabsLineTrigger value="disabled" disabled>
            Disabled
          </TabsLineTrigger>
          <TabsLineTrigger value="enabled">Enabled</TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent value="available">
          <div className="p-4 text-sm">
            This tab is available and can be clicked.
          </div>
        </TabsLineContent>
        <TabsLineContent value="disabled">
          <div className="p-4 text-sm">
            This tab is disabled and cannot be accessed.
          </div>
        </TabsLineContent>
        <TabsLineContent value="enabled">
          <div className="p-4 text-sm">
            This tab is also enabled and functional.
          </div>
        </TabsLineContent>
      </TabsLine>
    </div>
  ),
};

export const FullWidth: Story = {
  render: () => (
    <div className="p-8">
      <TabsLine defaultValue="tab1" className="w-full">
        <TabsLineList className="grid w-full grid-cols-3">
          <TabsLineTrigger value="tab1">Tab One</TabsLineTrigger>
          <TabsLineTrigger value="tab2">Tab Two</TabsLineTrigger>
          <TabsLineTrigger value="tab3">Tab Three</TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent value="tab1">
          <div className="p-4 text-sm">
            Content for the first tab with full width layout.
          </div>
        </TabsLineContent>
        <TabsLineContent value="tab2">
          <div className="p-4 text-sm">
            Content for the second tab with full width layout.
          </div>
        </TabsLineContent>
        <TabsLineContent value="tab3">
          <div className="p-4 text-sm">
            Content for the third tab with full width layout.
          </div>
        </TabsLineContent>
      </TabsLine>
    </div>
  ),
};
