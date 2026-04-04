import { Navigate } from 'react-router';
import { useEffect } from 'react';

export function Auth() {
  useEffect(() => {
    // Force set the bypass user in local storage
    localStorage.setItem('sambhav_user', JSON.stringify({
      email: 'admin@sambhav.ai',
      tier: 'power',
      user_id: '00000000-0000-0000-0000-000000000001',
    }));
  }, []);

  return <Navigate to="/dashboard" replace />;
}
