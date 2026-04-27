"use client";

import type { User } from "@supabase/supabase-js";
import { PencilSimpleIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
} from "@/components/molecules/Form/Form";

import { EASE_OUT } from "../../helpers";
import { useAccountCard } from "./useAccountCard";

interface Props {
  user: User;
  index?: number;
}

export function AccountCard({ user, index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const [emailDialogOpen, setEmailDialogOpen] = useState(false);
  const { emailForm, onSubmitEmail, isUpdatingEmail, currentEmail } =
    useAccountCard({ user });

  const watchedEmail = emailForm.watch("email");
  const emailHasError = Object.keys(emailForm.formState.errors).length > 0;
  const isSameEmail = watchedEmail === currentEmail;
  const disableEmailSubmit = emailHasError || isSameEmail;

  async function handleEmailSubmit(values: { email: string }) {
    if (disableEmailSubmit) return;
    const didUpdate = await onSubmitEmail(values);
    if (didUpdate) {
      setEmailDialogOpen(false);
    }
  }

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : {
              duration: 0.32,
              ease: EASE_OUT,
              delay: 0.04 + index * 0.05,
            }
      }
      className="flex w-full flex-col gap-2"
    >
      <div className="flex flex-col gap-1 px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Account
        </Text>
        <Text variant="small" className="text-zinc-500">
          Manage your sign-in details.
        </Text>
      </div>

      <div className="flex flex-col divide-y divide-zinc-200 rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex items-center justify-between gap-4 px-4 py-4">
          <Text variant="body-medium" as="span" className="text-textBlack">
            Email
          </Text>

          <div className="flex min-w-0 items-center gap-3">
            <Text
              variant="body"
              as="span"
              className="min-w-0 truncate text-textBlack"
            >
              {currentEmail}
            </Text>

            <Dialog
              title="Update email"
              styling={{ maxWidth: "420px" }}
              controlled={{
                isOpen: emailDialogOpen,
                set: (open) => {
                  setEmailDialogOpen(open);
                  if (open) emailForm.reset({ email: currentEmail });
                },
              }}
            >
              <Dialog.Trigger>
                <Button
                  variant="secondary"
                  size="small"
                  aria-label="Edit email"
                  className="h-7 min-w-0 px-1.5 py-0.5"
                >
                  <PencilSimpleIcon size={14} weight="duotone" />
                </Button>
              </Dialog.Trigger>
              <Dialog.Content>
                <Form form={emailForm} onSubmit={handleEmailSubmit}>
                  <div className="flex flex-col gap-4">
                    <Text variant="small" as="span" className="text-zinc-500">
                      We&apos;ll send a confirmation link to verify your new
                      address.
                    </Text>
                    <FormField
                      control={emailForm.control}
                      name="email"
                      render={({ field, fieldState }) => (
                        <FormItem className="space-y-0">
                          <FormControl>
                            <Input
                              id={field.name}
                              label="Email"
                              type="email"
                              autoComplete="email"
                              size="medium"
                              className="w-full"
                              wrapperClassName="!mb-0"
                              error={fieldState.error?.message}
                              {...field}
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                  </div>
                  <Dialog.Footer>
                    <Button
                      type="button"
                      variant="ghost"
                      size="small"
                      onClick={() => setEmailDialogOpen(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      size="small"
                      disabled={disableEmailSubmit}
                      loading={isUpdatingEmail}
                    >
                      {isUpdatingEmail ? "Saving" : "Update email"}
                    </Button>
                  </Dialog.Footer>
                </Form>
              </Dialog.Content>
            </Dialog>
          </div>
        </div>

        <div className="flex items-center justify-between gap-4 px-4 py-4">
          <Text variant="body-medium" as="span" className="text-textBlack">
            Password
          </Text>

          <Button
            as="NextLink"
            href="/reset-password"
            size="small"
            variant="secondary"
          >
            Reset password
          </Button>
        </div>
      </div>
    </motion.section>
  );
}
