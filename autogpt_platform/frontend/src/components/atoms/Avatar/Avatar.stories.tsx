import type { Meta, StoryObj } from "@storybook/nextjs";
import React from "react";
import {
  Avatar,
  AvatarImage,
  AvatarFallback,
} from "@/components/atoms/Avatar/Avatar";

const meta: Meta<typeof Avatar> = {
  title: "Atoms/Avatar",
  component: Avatar,
};

export default meta;

type Story = StoryObj<typeof Avatar>;

export const WithNextImage: Story = {
  render: () => (
    <div style={{ display: "flex", gap: 16 }}>
      <Avatar className="h-16 w-16">
        <AvatarImage
          as="NextImage"
          src="https://avatars.githubusercontent.com/u/9919?v=4"
          alt="GitHub Avatar"
        />
        <AvatarFallback>G</AvatarFallback>
      </Avatar>

      <Avatar className="h-10 w-10">
        <AvatarImage
          as="NextImage"
          src="https://avatars.githubusercontent.com/u/583231?v=4"
          alt="Octocat"
        />
        <AvatarFallback>O</AvatarFallback>
      </Avatar>
    </div>
  ),
};

export const WithImgTag: Story = {
  render: () => (
    <div style={{ display: "flex", gap: 16 }}>
      <Avatar className="h-16 w-16">
        <AvatarImage
          as="img"
          src="https://avatars.githubusercontent.com/u/139426?v=4"
          alt="User"
        />
        <AvatarFallback>U</AvatarFallback>
      </Avatar>

      <Avatar className="h-10 w-10">
        <AvatarImage as="img" src="" alt="No Image" />
        <AvatarFallback>N</AvatarFallback>
      </Avatar>
    </div>
  ),
};
