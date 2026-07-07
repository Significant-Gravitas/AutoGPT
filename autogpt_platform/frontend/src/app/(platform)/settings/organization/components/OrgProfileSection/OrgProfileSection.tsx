"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { useOrgProfileSection } from "./useOrgProfileSection";

interface Props {
  org: OrgResponse;
  isAdmin: boolean;
  onSaved: () => void;
}

export function OrgProfileSection({ org, isAdmin, onSaved }: Props) {
  const { form, isPending, isDirty, handleSubmit } = useOrgProfileSection({
    org,
    onSaved,
  });

  return (
    <section className="flex flex-col gap-4" data-testid="org-profile-section">
      <Text variant="h4" as="h2">
        Profile
      </Text>
      <Form
        form={form}
        onSubmit={handleSubmit}
        className="flex max-w-xl flex-col gap-4"
      >
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Input
                  {...field}
                  id={field.name}
                  label="Name"
                  disabled={!isAdmin}
                  wrapperClassName="!mb-0"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="slug"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Input
                  {...field}
                  id={field.name}
                  label="URL slug"
                  disabled={!isAdmin}
                  wrapperClassName="!mb-0"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="description"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Input
                  {...field}
                  id={field.name}
                  label="Description"
                  disabled={!isAdmin}
                  wrapperClassName="!mb-0"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        {isAdmin ? (
          <div>
            <Button type="submit" loading={isPending} disabled={!isDirty}>
              Save changes
            </Button>
          </div>
        ) : null}
      </Form>
    </section>
  );
}
