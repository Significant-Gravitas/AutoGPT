"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/molecules/Form/Form";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";

import { assignedTeamLabels } from "./helpers";
import { useInvitationsSection } from "./useInvitationsSection";

interface Props {
  orgId: string;
  isAdmin: boolean;
}

export function InvitationsSection({ orgId, isAdmin }: Props) {
  const {
    form,
    invitations,
    assignableTeams,
    teamNameById,
    isInviting,
    isRevoking,
    handleInvite,
    handleRevoke,
  } = useInvitationsSection({ orgId, isAdmin });

  if (!isAdmin) {
    return null;
  }

  return (
    <section
      className="flex flex-col gap-4"
      data-testid="org-invitations-section"
    >
      <Text variant="h4" as="h2">
        Invitations
      </Text>

      <Form
        form={form}
        onSubmit={handleInvite}
        className="flex max-w-xl flex-col gap-3"
      >
        <div className="flex items-start gap-3">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem className="flex-1">
                <FormControl>
                  <Input
                    {...field}
                    id={field.name}
                    label=""
                    hideLabel
                    placeholder="teammate@example.com"
                    wrapperClassName="!mb-0"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="isAdmin"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <label className="flex h-[2.875rem] items-center gap-2 text-sm text-zinc-600">
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                    Admin
                  </label>
                </FormControl>
              </FormItem>
            )}
          />
          <Button type="submit" loading={isInviting}>
            Invite
          </Button>
        </div>

        {assignableTeams.length > 0 ? (
          <FormField
            control={form.control}
            name="teamIds"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-xs text-zinc-500">
                  Pre-assign to teams
                </FormLabel>
                <FormControl>
                  <MultiToggle
                    aria-label="Pre-assign to teams"
                    items={assignableTeams.map((team) => ({
                      value: team.id,
                      label: team.name,
                    }))}
                    selectedValues={field.value}
                    onChange={field.onChange}
                  />
                </FormControl>
              </FormItem>
            )}
          />
        ) : null}
      </Form>

      {invitations.length > 0 ? (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {invitations.map((invitation) => (
            <li
              key={invitation.id}
              className="flex items-center gap-3 py-3"
              data-testid="org-invitation-row"
            >
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="truncate text-sm font-medium">
                  {invitation.email}
                </span>
                {invitation.team_ids.length > 0 ? (
                  <div className="flex flex-wrap items-center gap-1">
                    {assignedTeamLabels(invitation.team_ids, teamNameById).map(
                      (label) => (
                        <Badge key={label} variant="info">
                          {label}
                        </Badge>
                      ),
                    )}
                  </div>
                ) : null}
                <span className="truncate text-xs text-zinc-500">
                  Expires {new Date(invitation.expires_at).toLocaleDateString()}
                </span>
              </div>
              {invitation.is_admin ? <Badge variant="info">Admin</Badge> : null}
              <Button
                variant="ghost"
                size="small"
                loading={isRevoking}
                onClick={() => handleRevoke(invitation)}
              >
                Revoke
              </Button>
            </li>
          ))}
        </ul>
      ) : (
        <Text variant="small" className="text-zinc-500">
          No pending invitations.
        </Text>
      )}
    </section>
  );
}
