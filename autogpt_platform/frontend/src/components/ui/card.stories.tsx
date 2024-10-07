import type { Meta, StoryObj } from '@storybook/react';

import { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent } from './card';

const meta = {
  title: 'UI/Card',
  component: Card,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    // Add any specific controls for Card props here if needed
  },
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: (
      <>
        <CardHeader>
          <CardTitle>Card Title</CardTitle>
          <CardDescription>Card Description</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Card Content</p>
        </CardContent>
        <CardFooter>
          <p>Card Footer</p>
        </CardFooter>
      </>
    ),
  },
};

export const HeaderOnly: Story = {
  args: {
    children: (
      <CardHeader>
        <CardTitle>Header Only Card</CardTitle>
        <CardDescription>This card has only a header.</CardDescription>
      </CardHeader>
    ),
  },
};

export const ContentOnly: Story = {
  args: {
    children: (
      <CardContent>
        <p>This card has only content.</p>
      </CardContent>
    ),
  },
};

export const FooterOnly: Story = {
  args: {
    children: (
      <CardFooter>
        <p>This card has only a footer.</p>
      </CardFooter>
    ),
  },
};

export const CustomContent: Story = {
  args: {
    children: (
      <>
        <CardHeader>
          <CardTitle>Custom Content</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-40 bg-gray-100 rounded-md">
            <span role="img" aria-label="Rocket" style={{ fontSize: '3rem' }}>🚀</span>
          </div>
        </CardContent>
        <CardFooter className="justify-between">
          <button className="px-4 py-2 bg-blue-500 text-white rounded">Action</button>
          <p>Footer text</p>
        </CardFooter>
      </>
    ),
  },
};
