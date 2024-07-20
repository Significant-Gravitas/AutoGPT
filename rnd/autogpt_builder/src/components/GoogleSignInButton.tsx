import React, { useEffect, useCallback } from 'react';
import { Button } from './ui/button';
import { useGoogleAuth } from '@/hooks/useGoogleAuth';

interface GoogleSignInButtonProps {
  onTokenChange: (token: string | null) => void;
}

const GoogleSignInButton: React.FC<GoogleSignInButtonProps> = React.memo(({ onTokenChange }) => {
  const { token, error, login, logout } = useGoogleAuth('https://www.googleapis.com/auth/spreadsheets.readonly');

  const handleTokenChange = useCallback((newToken: string | null) => {
    onTokenChange(newToken);
  }, [onTokenChange]);

  useEffect(() => {
    handleTokenChange(token);
  }, [token, handleTokenChange]);

  if (error) {
    return <div>Error: {error}</div>;
  }

  return token ? (
    <Button onClick={logout}>Sign Out of Google Sheets</Button>
  ) : (
    <Button onClick={() => login()}>Sign in with Google Sheets</Button>
  );
});

GoogleSignInButton.displayName = 'GoogleSignInButton';

export default GoogleSignInButton;