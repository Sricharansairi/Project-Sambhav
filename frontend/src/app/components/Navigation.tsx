import { motion } from 'motion/react';
import { Link, useLocation, useNavigate } from 'react-router';
import { Home, BarChart3, History, Target, User, Info, Menu, X, SlidersHorizontal, LogOut } from 'lucide-react';
import { useState, useEffect } from 'react';
import { AnimatedIcon } from './AnimatedIcon';
import logoImage from '../../assets/066d6bda782cfe271b2a192b0848783b83987f2e.png';
import { getToken, clearToken } from '../lib/api';
import { sounds } from '../lib/audio';

const navItems = [
  { path: '/',            label: 'Home',       icon: Home },
  { path: '/dashboard',   label: 'Predict',    icon: BarChart3 },
  { path: '/fact-check',  label: 'Fact Check', icon: Target },
  { path: '/calibration', label: 'Calibrate',  icon: SlidersHorizontal },
  { path: '/about',       label: 'About',      icon: Info },
];

export function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    // Check if token exists on mount and when location changes
    setIsLoggedIn(!!getToken());
  }, [location.pathname]);

  const handleLogout = () => {
    sounds.click();
    clearToken();
    localStorage.removeItem('sambhav_user');
    setIsLoggedIn(false);
    navigate('/auth');
  };

  return (
    <>
      {/* Desktop Navigation */}
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="fixed top-0 left-0 right-0 z-50 px-4 py-3"
      >
        <div className="max-w-7xl mx-auto">
          <div
            className="backdrop-blur-xl bg-white/[0.03] border border-white/10 rounded-xl px-4 py-2"
            style={{
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 0 20px rgba(255, 255, 255, 0.02)',
            }}
          >
            <div className="flex items-center justify-between">
              {/* Logo + Title */}
              <Link to="/" className="flex items-center gap-2 group">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  transition={{ duration: 0.3 }}
                  className="w-8 h-8"
                >
                  <img src={logoImage} alt="Sambhav" className="w-full h-full object-contain" />
                </motion.div>
                <div>
                  <h1 className="text-base font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent leading-tight">
                    SAMBHAV
                  </h1>
                  {/* "Uncertainty, Quantified" line */}
                  <p className="text-[10px] text-muted-foreground hidden sm:block leading-tight">
                    Uncertainty, Quantified
                  </p>
                  {/* NEW: Multi-Modal subtitle — same font-family, same muted color, smaller */}
                  <p className="text-[9px] text-muted-foreground/60 hidden sm:block leading-tight tracking-wide">
                    A Multi-Modal Probabilistic Inference Engine
                  </p>
                </div>
              </Link>

              {/* Desktop Nav Items */}
              <div className="hidden md:flex items-center gap-1">
                {navItems.map((item) => {
                  const isActive = location.pathname === item.path;
                  return (
                    <Link key={item.path} to={item.path} className="relative" onClick={() => sounds.click()}>
                      <motion.div
                        className={`
                          px-3 py-1.5 rounded-lg flex items-center gap-1.5
                          transition-colors duration-200
                          ${isActive
                            ? 'text-primary'
                            : 'text-muted-foreground hover:text-foreground'
                          }
                        `}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <AnimatedIcon icon={item.icon} size={16} hoverEffect="twitch" />
                        <span className="text-sm">{item.label}</span>
                      </motion.div>
                      {isActive && (
                        <motion.div
                          layoutId="activeTab"
                          className="absolute inset-0 bg-primary/10 rounded-lg border border-primary/30"
                          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                        />
                      )}
                    </Link>
                  );
                })}
                {isLoggedIn ? (
                  <motion.button
                    onClick={handleLogout}
                    className="ml-3 px-3 py-1.5 text-sm rounded-lg bg-white/5 border border-white/10 text-destructive font-medium flex items-center gap-1.5 hover:bg-destructive/10"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <LogOut size={14} />
                    <span>Logout</span>
                  </motion.button>
                ) : (
                  <Link to="/auth">
                    <motion.button
                      className="ml-3 px-3 py-1.5 text-sm rounded-lg bg-primary text-black font-medium"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Sign In
                    </motion.button>
                  </Link>
                )}
              </div>

              {/* Mobile Menu Button */}
              <button
                className="md:hidden p-2"
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              >
                {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
              </button>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="fixed top-16 left-4 right-4 z-40 md:hidden"
        >
          <div
            className="backdrop-blur-xl bg-white/[0.05] border border-white/10 rounded-xl p-3"
            style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)' }}
          >
            <div className="flex flex-col gap-1.5">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`
                      px-3 py-2 text-sm rounded-lg flex items-center gap-2
                      ${isActive
                        ? 'bg-primary/10 text-primary border border-primary/30'
                        : 'text-muted-foreground hover:bg-white/5'
                      }
                    `}
                  >
                    <item.icon size={16} />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
              {isLoggedIn ? (
                <button
                  onClick={handleLogout}
                  className="px-3 py-2 text-sm rounded-lg flex items-center gap-2 text-destructive hover:bg-destructive/10"
                >
                  <LogOut size={16} />
                  <span>Logout</span>
                </button>
              ) : (
                <Link
                  to="/auth"
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="px-3 py-2 text-sm rounded-lg flex items-center gap-2 text-primary hover:bg-primary/10"
                >
                  <User size={16} />
                  <span>Sign In</span>
                </Link>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </>
  );
}
