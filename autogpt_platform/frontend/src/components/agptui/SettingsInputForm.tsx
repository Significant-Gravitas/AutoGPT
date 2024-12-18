"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import useSupabase from "@/hooks/useSupabase";

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
  const [password, setPassword] = React.useState("");
  const [confirmPassword, setConfirmPassword] = React.useState("");
  const [passwordsMatch, setPasswordsMatch] = React.useState(true);
  const { supabase } = useSupabase();

  const handleSaveChanges = async () => {
    if (password !== confirmPassword) {
      setPasswordsMatch(false);
      return;
    }
    setPasswordsMatch(true);
    if (supabase) {
      try {
        const { error } = await supabase.auth.updateUser({
          password: password,
        });
        if (error) {
          console.error("Error updating user:", error);
        } else {
          console.log("User updated successfully");
        }
      } catch (error) {
        console.error("Error updating user:", error);
      }
    }
  };

  const handleCancel = () => {
    setPassword("");
    setConfirmPassword("");
    setPasswordsMatch(true);
  };

  return (
    <div className="mx-auto w-full max-w-[1077px] bg-white px-4 pt-8 dark:bg-neutral-900 sm:px-6 sm:pt-16">
      <h1 className="mb-8 text-2xl font-semibold text-slate-950 dark:text-slate-200 sm:mb-16 sm:text-3xl">
        Settings
      </h1>

      {/* My Account Section */}
      <section aria-labelledby="account-heading">
        <h2
          id="account-heading"
          className="mb-8 text-lg font-medium text-neutral-500 dark:text-neutral-400 sm:mb-12"
        >
          My account
        </h2>
        <div className="flex max-w-[800px] flex-col gap-7">
          {/* Password Input */}
          <div className="relative">
            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="password-input"
                className="text-base font-medium text-slate-950 dark:text-slate-200"
              >
                Password
              </label>
              <input
                id="password-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="h-[50px] w-full rounded-[35px] border border-neutral-200 bg-transparent px-6 py-3 text-base text-slate-950 dark:border-neutral-700 dark:text-white"
                aria-label="Password field"
              />
            </div>
          </div>

          {/* Confirm Password Input */}
          <div className="relative">
            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="confirm-password-input"
                className="text-base font-medium text-slate-950 dark:text-slate-200"
              >
                Confirm Password
              </label>
              <input
                id="confirm-password-input"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="h-[50px] w-full rounded-[35px] border border-neutral-200 bg-transparent px-6 py-3 text-base text-slate-950 dark:border-neutral-700 dark:text-white"
                aria-label="Confirm Password field"
              />
            </div>
          </div>
        </div>
      </section>

      <div
        className="my-8 border-t border-neutral-200 dark:border-neutral-700 sm:my-12"
        role="separator"
      />

      <div className="mt-8 flex justify-end">
        <div className="flex gap-3">
          <Button
            variant="secondary"
            className="h-[50px] rounded-[35px] bg-neutral-200 px-6 py-3 font-['Geist'] text-base font-medium text-neutral-800 transition-colors hover:bg-neutral-300 dark:bg-neutral-700 dark:text-neutral-200 dark:hover:bg-neutral-600"
            onClick={handleCancel}
          >
            Cancel
          </Button>
          <Button
            variant="default"
            className="h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 font-['Geist'] text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-900 dark:hover:bg-neutral-800"
            onClick={handleSaveChanges}
          >
            Save changes
          </Button>
        </div>
      </div>
    </div>
  );
};
