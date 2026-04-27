"use client";

import type { User } from "@supabase/supabase-js";
import { ShieldCheckIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
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
  const { form, onSubmit, isLoading, currentEmail } = useAccountCard({ user });

  const watched = form.watch("email");
  const hasError = Object.keys(form.formState.errors).length > 0;
  const isSameEmail = watched === currentEmail;
  const disableSubmit = hasError || isSameEmail;

  return (
    <motion.section
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }}
      className="rounded-[18px] border border-zinc-200 bg-white p-6 shadow-[0_1px_2px_rgba(15,15,20,0.04)] transition-shadow duration-200 ease-out focus-within:shadow-[0_8px_28px_-12px_rgba(15,15,20,0.12)]"
    >
      <CardTitle
        eyebrow="Account & Security"
        title="How you sign in"
        description="Update the email tied to your account or rotate your password."
      />

      <Form form={form} onSubmit={onSubmit} className="mt-6 space-y-0">
        <FormField
          control={form.control}
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
                  error={fieldState.error?.message}
                  {...field}
                />
              </FormControl>
            </FormItem>
          )}
        />

        <div className="-mt-2 flex flex-col-reverse items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 text-zinc-500">
            <ShieldCheckIcon size={16} weight="duotone" />
            <Text variant="small" as="span" className="text-zinc-500">
              We'll send a confirmation link to verify your new address.
            </Text>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              as="NextLink"
              href="/reset-password"
              size="small"
            >
              Reset password
            </Button>
            <Button
              type="submit"
              size="small"
              disabled={disableSubmit}
              loading={isLoading}
            >
              {isLoading ? "Saving" : "Update email"}
            </Button>
          </div>
        </div>
      </Form>
    </motion.section>
  );
}

function CardTitle({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Text
        variant="small-medium"
        as="span"
        className="uppercase tracking-[0.08em] text-zinc-400"
      >
        {eyebrow}
      </Text>
      <Text variant="h4" as="h2" className="text-[#1F1F20]">
        {title}
      </Text>
      <Text variant="small" className="text-zinc-500">
        {description}
      </Text>
    </div>
  );
}
