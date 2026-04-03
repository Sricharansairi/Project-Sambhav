import { useEffect } from 'react';
import { useNavigate } from 'react-router';

export function Dashboard() {
  const navigate = useNavigate();

  useEffect(() => {
    navigate('/mode-selection');
  }, [navigate]);

  return null;
}