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
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <input
                  id="email-input"
                  type="email"
                  value={email}
                  className="w-full rounded-full border border-neutral-200 bg-transparent px-4 py-2.5 text-base sm:w-[638px]"
                  readOnly
                  aria-label="Email address"
                />
                <div className="w-full sm:ml-4 sm:w-[88px]">
                  <Button
                    variant="default"
                    size="sm"
                    className="h-[43px] w-full rounded-full bg-black text-white hover:bg-black/90"
                    aria-label="Edit email"
                  >
                    Edit
                  </Button>
                </div>
              </div>
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
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <input
                  id="password-input"
                  type="password"
                  value="************"
                  className="w-full rounded-full border border-neutral-200 bg-transparent px-4 py-2.5 text-base sm:w-[638px]"
                  readOnly
                  aria-label="Password field"
                />
                <div className="w-full sm:ml-4 sm:w-[88px]">
                  <Button
                    variant="default"
                    size="sm"
                    className="h-[43px] w-full rounded-full bg-black text-white hover:bg-black/90"
                    aria-label="Edit password"
                  >
                    Edit
                  </Button>
                </div>
              </div>
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
    </div>
  );
};
