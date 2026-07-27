import { UserIcon } from "@phosphor-icons/react";

interface Props {
  name: string;
  avatarUrl: string | null;
  size?: "default" | "small";
}

export function ExpertAvatar({ name, avatarUrl, size = "default" }: Props) {
  const dimension = size === "small" ? "h-4 w-4" : "h-6 w-6";
  if (avatarUrl) {
    return (
      <img
        src={avatarUrl}
        alt={name}
        className={`${dimension} rounded-full object-cover`}
      />
    );
  }
  return (
    <div
      className={`${dimension} flex items-center justify-center rounded-full bg-purple-100 text-purple-700`}
    >
      <UserIcon
        className={size === "small" ? "h-2.5 w-2.5" : "h-3.5 w-3.5"}
        weight="fill"
      />
    </div>
  );
}
