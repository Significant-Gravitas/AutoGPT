import type { Meta, StoryObj } from '@storybook/react';
import { NavBar } from './NavBar';
import { UserProvider } from '@/context/UserContext'; // You'll need to create this context

const meta: Meta<typeof NavBar> = {
  title: 'Components/NavBar',
  component: NavBar,
  parameters: {
    layout: 'fullscreen',
  },
  decorators: [
    (Story) => (
      <UserProvider>
        <Story />
      </UserProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof NavBar>;

export const Default: Story = {
  render: () => <NavBar />,
};

// You might need to mock the server-side functionality
// for the story to work properly in Storybook
export const LoggedOut: Story = {
  parameters: {
    userContext: { user: null, isAvailable: true },
  },
};

export const LoggedIn: Story = {
  parameters: {
    userContext: { 
      user: { id: '1', name: 'John Doe', email: 'john@example.com' },
      isAvailable: true
    },
  },
};
