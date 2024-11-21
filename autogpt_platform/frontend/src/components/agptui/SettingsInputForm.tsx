"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

interface SettingsInputFormProps {
  email?: string;
  desktopNotifications?: {
    first: boolean;
    second: boolean;
  };
}

export const SettingsInputForm = ({
  email = "johndoe@email.com",
  desktopNotifications = { first: false, second: true },
}: SettingsInputFormProps) => {
  const [notifications, setNotifications] =
    React.useState(desktopNotifications);

  const handleToggleFirst = () => {
    setNotifications((prev) => ({
      ...prev,
      first: !prev.first,
    }));
  };

  const handleToggleSecond = () => {
    setNotifications((prev) => ({
      ...prev,
      second: !prev.second,
    }));
  };

  return (
    <div className="mx-auto w-full max-w-[1077px] px-4 pt-8 sm:px-6 sm:pt-16">
      <h1 className="mb-8 text-2xl font-semibold sm:mb-16 sm:text-3xl">
        Settings
      </h1>

      {/* My Account Section */}
      <section aria-labelledby="account-heading">
        <h2
          id="account-heading"
          className="mb-8 text-lg font-medium text-neutral-500 sm:mb-12"
        >
          My account
        </h2>
        <div className="flex flex-col gap-7">
          <div className="relative">
            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="email-input"
                className="text-base font-medium text-slate-950"
              >
                Email
              </label>
              <input
                id="email-input"
                type="email"
                value={email}
                className="w-full rounded-full border border-neutral-200 bg-transparent px-4 py-2.5 text-base"
                readOnly
                aria-label="Email address"
              />
            </div>
          </div>

          <div className="relative">
            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="password-input"
                className="text-base font-medium text-slate-950"
              >
                Password
              </label>
              <input
                id="password-input"
                type="password"
                value="************"
                className="w-full rounded-full border border-neutral-200 bg-transparent px-4 py-2.5 text-base"
                readOnly
                aria-label="Password field"
              />
            </div>
          </div>
        </div>
      </section>

      <div
        className="my-8 border-t border-neutral-200 sm:my-12"
        role="separator"
      />

      {/* Notifications Section */}
      <section aria-labelledby="notifications-heading">
        <h2
          id="notifications-heading"
          className="mb-8 text-lg font-medium text-neutral-500 sm:mb-12"
        >
          Notifications
        </h2>
        <div className="flex flex-col gap-7">
          <div className="flex flex-col gap-4 sm:flex-row">
            <div className="w-full sm:w-[638px]">
              <h3
                id="desktop-notif-1"
                className="text-base font-medium text-slate-950"
              >
                Enable desktop notifications
              </h3>
              <p className="mt-2 text-base text-neutral-600">
                More detailed explanation for the notifications that this person
                is enabling
              </p>
            </div>
            <div className="flex h-[43px] items-center sm:ml-4 sm:w-[88px] sm:justify-center">
              <Switch
                checked={notifications.first}
                onCheckedChange={handleToggleFirst}
                aria-labelledby="desktop-notif-1"
                aria-label="Toggle desktop notifications"
              />
            </div>
          </div>

          <div className="flex flex-col gap-4 sm:flex-row">
            <div className="w-full sm:w-[638px]">
              <h3
                id="desktop-notif-2"
                className="text-base font-medium text-slate-950"
              >
                Enable desktop notifications
              </h3>
              <p className="mt-2 text-base text-neutral-600">
                More detailed explanation for the notifications that this person
                is enabling
              </p>
            </div>
            <div className="flex h-[43px] items-center sm:ml-4 sm:w-[88px] sm:justify-center">
              <Switch
                checked={notifications.second}
                onCheckedChange={handleToggleSecond}
                aria-labelledby="desktop-notif-2"
                aria-label="Toggle desktop notifications"
              />
            </div>
          </div>
        </div>
      </section>

      <div className="mt-8 flex justify-end">
        <div className="flex gap-3">
          <Button
            variant="secondary"
            className="h-[50px] rounded-[35px] bg-neutral-200 px-6 py-3 font-['Geist'] text-base font-medium text-neutral-800 transition-colors hover:bg-neutral-300"
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            className="h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 font-['Geist'] text-base font-medium text-white transition-colors hover:bg-neutral-900"
          >
            Save changes
          </Button>
        </div>
      </div>
    </div>
  );
};
