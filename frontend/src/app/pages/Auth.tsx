import { motion, AnimatePresence } from 'motion/react';
import { Link, useNavigate } from 'react-router';
import { Lock, Mail, ArrowRight, User, Eye, EyeOff, UserCircle, KeyRound, AlertTriangle, CheckCircle } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { GlassCard } from '../components/GlassCard';
import { useState } from 'react';
import logoImage from '../../assets/066d6bda782cfe271b2a192b0848783b83987f2e.png';

import { auth } from '../lib/api';

type AuthView = 'signin' | 'signup' | 'forgot' | 'reset';

export function Auth() {
  const navigate = useNavigate();
  const [view, setView] = useState<AuthView>('signin');
  const [showPassword, setShowPassword] = useState(false);

  // Form fields
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [remember, setRemember] = useState(false);
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  // State
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const isSignUp = view === 'signup';

  const validateEmail = (email: string) => {
    return String(email)
      .toLowerCase()
      .match(
        /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
      );
  };

  // === Sign In ===
  const handleSignIn = async () => {
    if (!email || !password) { setError('Email and password are required'); return; }
    if (!validateEmail(email)) { setError('Please enter a valid email address'); return; }
    setIsLoading(true); setError(null);
    try {
      const data = await auth.login(email, password);
      if (data.success) {
        localStorage.setItem('sambhav_user', JSON.stringify({
          email: data.email || email,
          tier: data.tier || 'free',
          user_id: data.user_id || '',
        }));
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Invalid email or password');
      }
    } catch (e: any) {
      setError(e.message || 'Failed to connect to server. Please check if the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  // === Sign Up ===
  const handleSignUp = async () => {
    if (!email || !password) { setError('Email and password are required'); return; }
    if (!validateEmail(email)) { setError('Please enter a valid email address'); return; }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return; }
    if (!agreedToTerms) { setError('You must agree to the Terms of Service and Privacy Policy'); return; }
    setIsLoading(true); setError(null);
    try {
      const data = await auth.register(email, password);
      if (data.success) {
        localStorage.setItem('sambhav_user', JSON.stringify({
          email: data.email || email,
          tier: data.tier || 'free',
          user_id: data.user_id || '',
        }));
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Registration failed');
      }
    } catch (e: any) {
      setError(e.message || 'Failed to connect to server.');
    } finally {
      setIsLoading(false);
    }
  };

  // === Guest Mode ===
  const handleGuest = async () => {
    setIsLoading(true); setError(null);
    try {
      // Use the new /auth/guest endpoint for a secure token
      const res = await fetch(`${import.meta.env.VITE_API_URL || ''}/api/auth/guest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await res.json();
      if (data.token) {
        auth.setToken(data.token);
        auth.setUser({ email: 'guest', tier: 'guest' });
        navigate('/dashboard');
      } else {
        throw new Error('Guest login failed');
      }
    } catch (err: any) {
      setError(err.message || 'Guest entry failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // === Password Reset (Forgot flow) ===
  const handleResetPassword = async () => {
    if (!email) { setError('Please enter your email address'); return; }
    if (!validateEmail(email)) { setError('Please enter a valid email address'); return; }
    if (!newPassword || newPassword.length < 6) { setError('New password must be at least 6 characters'); return; }
    if (newPassword !== confirmPassword) { setError('Passwords do not match'); return; }
    setIsLoading(true); setError(null);
    try {
      // Simulation of sending a reset email
      console.log(`Sending reset link to ${email}...`);
      
      const data = await auth.resetPassword(email, newPassword);
      if (data.success) {
        setSuccessMsg('A password reset link has been sent to your email (simulated). Password updated!');
        setTimeout(() => {
          setView('signin');
          setSuccessMsg(null);
          setPassword('');
          setNewPassword('');
          setConfirmPassword('');
        }, 3000);
      } else {
        setError(data.detail || 'Password reset failed');
      }
    } catch (e: any) {
      setError(e.message || 'Failed to connect to server.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (view === 'signin') handleSignIn();
    else if (view === 'signup') handleSignUp();
    else if (view === 'forgot' || view === 'reset') handleResetPassword();
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      
      <div className="relative z-10 min-h-screen grid md:grid-cols-5">
        {/* Left: Branding (40%) */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="md:col-span-2 flex flex-col justify-center p-8 md:p-12 bg-white/[0.02]"
        >
          <Link to="/" className="inline-flex items-center gap-3 mb-12">
            <img src={logoImage} alt="Sambhav" className="w-10 h-10" />
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                SAMBHAV
              </h1>
              <p className="text-xs text-muted-foreground">A Multi-Modal Probabilistic Inference</p>
            </div>
          </Link>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-3xl font-bold mb-4">
              {view === 'forgot' || view === 'reset' ? 'Reset Password' :
               isSignUp ? 'Join Sambhav' : 'Welcome Back'}
            </h2>
            <p className="text-base text-muted-foreground mb-6">
              {view === 'forgot' || view === 'reset' 
                ? 'Enter your email and set a new password to regain access.'
                : isSignUp 
                  ? 'Start your journey in probabilistic inference and uncertainty quantification.'
                  : 'Continue your journey in probabilistic inference and uncertainty quantification.'}
            </p>

            <div className="space-y-3">
              <div className="flex items-center gap-3 text-sm text-foreground/70">
                <div className="w-1 h-1 bg-primary rounded-full" />
                <span>Access 12 operating modes</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-foreground/70">
                <div className="w-1 h-1 bg-secondary rounded-full" />
                <span>Track prediction calibration</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-foreground/70">
                <div className="w-1 h-1 bg-accent rounded-full" />
                <span>Multi-domain analysis</span>
              </div>
            </div>
          </motion.div>
        </motion.div>

        {/* Right: Form (60%) */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="md:col-span-3 flex items-center justify-center p-6 md:p-12"
        >
          <GlassCard variant="elevated" className="w-full max-w-md p-6">
            {/* Tabs — only for signin/signup */}
            {(view === 'signin' || view === 'signup') && (
              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => { setView('signin'); setError(null); }}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    view === 'signin'
                      ? 'bg-primary text-black'
                      : 'bg-white/5 text-muted-foreground hover:bg-white/10'
                  }`}
                >
                  Sign In
                </button>
                <button
                  onClick={() => { setView('signup'); setError(null); }}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    view === 'signup'
                      ? 'bg-primary text-black'
                      : 'bg-white/5 text-muted-foreground hover:bg-white/10'
                  }`}
                >
                  Sign Up
                </button>
              </div>
            )}

            {/* Back to Sign In (for forgot/reset) */}
            {(view === 'forgot' || view === 'reset') && (
              <button
                onClick={() => { setView('signin'); setError(null); setSuccessMsg(null); }}
                className="text-sm text-primary hover:underline mb-4 flex items-center gap-1"
              >
                ← Back to Sign In
              </button>
            )}

            {/* Error / Success Messages */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mb-4 p-3 rounded-lg bg-destructive/10 border border-destructive/30 flex items-start gap-2"
                >
                  <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                  <p className="text-xs text-destructive">{error}</p>
                </motion.div>
              )}
              {successMsg && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mb-4 p-3 rounded-lg bg-success/10 border border-success/30 flex items-start gap-2"
                >
                  <CheckCircle className="w-4 h-4 text-success shrink-0 mt-0.5" />
                  <p className="text-xs text-success">{successMsg}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <form className="space-y-4" onSubmit={handleSubmit}>
              {/* Name (Sign Up only) */}
              {isSignUp && (
                <div>
                  <label className="block text-sm mb-1.5">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <input
                      type="text"
                      value={name}
                      onChange={e => setName(e.target.value)}
                      placeholder="John Doe"
                      className="w-full pl-10 pr-4 py-2.5 text-sm bg-white/5 border border-white/10 rounded-lg
                               focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                               transition-all placeholder:text-muted-foreground/50"
                    />
                  </div>
                </div>
              )}

              {/* Email */}
              <div>
                <label className="block text-sm mb-1.5">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="your.email@example.com"
                    className="w-full pl-10 pr-4 py-2.5 text-sm bg-white/5 border border-white/10 rounded-lg
                             focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                             transition-all placeholder:text-muted-foreground/50"
                  />
                </div>
              </div>

              {/* Password (Sign In / Sign Up) */}
              {(view === 'signin' || view === 'signup') && (
                <div>
                  <label className="block text-sm mb-1.5">Password</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      placeholder="••••••••"
                      className="w-full pl-10 pr-10 py-2.5 text-sm bg-white/5 border border-white/10 rounded-lg
                               focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                               transition-all placeholder:text-muted-foreground/50"
                    />
                    <button 
                      type="button" 
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              )}

              {/* New Password fields (Forgot/Reset) */}
              {(view === 'forgot' || view === 'reset') && (
                <>
                  <div>
                    <label className="block text-sm mb-1.5">New Password</label>
                    <div className="relative">
                      <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        type={showPassword ? 'text' : 'password'}
                        value={newPassword}
                        onChange={e => setNewPassword(e.target.value)}
                        placeholder="Min 6 characters"
                        className="w-full pl-10 pr-10 py-2.5 text-sm bg-white/5 border border-white/10 rounded-lg
                                 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                                 transition-all placeholder:text-muted-foreground/50"
                      />
                      <button 
                        type="button" 
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm mb-1.5">Confirm Password</label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        type="password"
                        value={confirmPassword}
                        onChange={e => setConfirmPassword(e.target.value)}
                        placeholder="Re-enter new password"
                        className="w-full pl-10 pr-4 py-2.5 text-sm bg-white/5 border border-white/10 rounded-lg
                                 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                                 transition-all placeholder:text-muted-foreground/50"
                      />
                    </div>
                  </div>
                </>
              )}

              {/* Remember / Forgot (Sign In only) */}
              {view === 'signin' && (
                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={remember}
                      onChange={e => setRemember(e.target.checked)}
                      className="w-3.5 h-3.5 rounded border-white/20" 
                    />
                    <span className="text-sm text-muted-foreground">Remember me</span>
                  </label>
                  <button 
                    type="button"
                    onClick={() => { setView('forgot'); setError(null); }}
                    className="text-sm text-primary hover:underline"
                  >
                    Forgot password?
                  </button>
                </div>
              )}

              {/* Terms (Sign Up only) */}
              {isSignUp && (
                <label className="flex items-start gap-2 cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={agreedToTerms}
                    onChange={e => setAgreedToTerms(e.target.checked)}
                    className="w-3.5 h-3.5 rounded border-white/20 mt-0.5" 
                  />
                  <span className="text-xs text-muted-foreground">
                    I agree to the Terms of Service and Privacy Policy
                  </span>
                </label>
              )}

              {/* Submit */}
              <motion.button
                type="submit"
                disabled={isLoading}
                className="w-full px-6 py-2.5 rounded-lg bg-primary text-black font-medium text-sm
                         flex items-center justify-center gap-2 group disabled:opacity-50"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isLoading ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-4 h-4 border-2 border-black border-t-transparent rounded-full"
                  />
                ) : (
                  <>
                    <span>
                      {view === 'signin' ? 'Sign In' : 
                       view === 'signup' ? 'Create Account' : 
                       'Reset Password'}
                    </span>
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </motion.button>
            </form>

            {/* Continue as Guest */}
            {(view === 'signin' || view === 'signup') && (
              <div className="mt-4">
                <div className="flex items-center gap-3 my-4">
                  <div className="flex-1 h-px bg-white/10" />
                  <span className="text-xs text-muted-foreground">or</span>
                  <div className="flex-1 h-px bg-white/10" />
                </div>
                <motion.button
                  onClick={handleGuest}
                  disabled={isLoading}
                  className="w-full px-6 py-2.5 rounded-lg bg-white/5 border border-white/10 text-foreground 
                           font-medium text-sm flex items-center justify-center gap-2 hover:bg-white/10 
                           transition-colors disabled:opacity-50"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <UserCircle className="w-4 h-4" />
                  <span>Continue as Guest</span>
                </motion.button>
              </div>
            )}
          </GlassCard>
        </motion.div>
      </div>
    </div>
  );
}
