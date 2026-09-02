import { TaskCredentialRef } from "@/app/api/__generated__/models/taskCredentialRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Key01Icon } from "@hugeicons/core-free-icons";
import { TaskCard } from "./TaskCard";

interface Props {
  credentials: TaskCredentialRef[];
}

/** The connections the task's runs were configured with — names only, never
 *  secret material. */
export function TaskCredentialsCard({ credentials }: Props) {
  if (credentials.length === 0) return null;

  return (
    <TaskCard title="Credentials used">
      <ul className="flex flex-col gap-2" aria-label="Credentials used">
        {credentials.map((credential) => (
          <li
            key={credential.id}
            className="flex items-center gap-2 rounded-xl bg-zinc-50 p-2.5"
          >
            <Icon
              icon={Key01Icon}
              size={16}
              className="shrink-0 text-zinc-400"
            />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-medium text-zinc-900">
                {credential.title || credential.provider}
              </span>
              {credential.title ? (
                <span className="block text-[11px] text-zinc-500">
                  {credential.provider}
                </span>
              ) : null}
            </span>
          </li>
        ))}
      </ul>
    </TaskCard>
  );
}
