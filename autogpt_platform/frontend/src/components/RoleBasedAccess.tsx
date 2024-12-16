// components/RoleBasedAccess.tsx
import React from "react";
import { useSupabase } from "./providers/SupabaseProvider";

interface RoleBasedAccessProps {
  allowedRoles: string[];
  children: React.ReactNode;
}

const RoleBasedAccess: React.FC<RoleBasedAccessProps> = ({
  allowedRoles,
  children,
}) => {
  const { user, isLoading } = useSupabase();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!user!.role || !allowedRoles.includes(user!.role)) {
    return null;
  }

  return <>{children}</>;
};

export default RoleBasedAccess;
