"use client";

import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";

import { JOIN_POLICY_OPTIONS } from "../CreateTeamDialog/schema";
import { useTeamProfileForm } from "./useTeamProfileForm";

interface Props {
  orgId: string;
  team: TeamResponse;
  onSaved: () => void;
}

export function TeamProfileForm({ orgId, team, onSaved }: Props) {
  const { form, isPending, isDirty, handleSubmit } = useTeamProfileForm({
    orgId,
    team,
    onSaved,
  });

  return (
    <Form
      form={form}
      onSubmit={handleSubmit}
      className="flex max-w-lg flex-col gap-4"
    >
      <FormField
        control={form.control}
        name="name"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id={`team-name-${team.id}`}
                label="Name"
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
                id={`team-description-${team.id}`}
                label="Description"
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <FormField
        control={form.control}
        name="join_policy"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Select
                id={`team-join-policy-${team.id}`}
                label="Join policy"
                value={field.value}
                onValueChange={field.onChange}
                options={JOIN_POLICY_OPTIONS}
                disabled={team.is_default}
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <div>
        <Button type="submit" loading={isPending} disabled={!isDirty}>
          Save changes
        </Button>
      </div>
    </Form>
  );
}
