import { Component, type ReactNode } from 'react';
import { RouterProvider } from 'react-router';
import { router } from './routes';
import { StickyDisclaimer } from './components/StickyDisclaimer';

// ── Global error boundary ─────────────────────────────────────
interface EBState { error: Error | null }

class ErrorBoundary extends Component<{ children: ReactNode }, EBState> {
  state: EBState = { error: null };

  static getDerivedStateFromError(error: Error): EBState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[Sambhav ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background px-6">
          <div className="max-w-lg w-full text-center space-y-6">
            {/* Header */}
            <div className="space-y-2">
              <p className="text-[11px] tracking-widest text-muted-foreground uppercase font-mono">
                Application Error
              </p>
              <h1 className="text-2xl font-bold text-foreground">
                Something went wrong
              </h1>
              <p className="text-sm text-muted-foreground">
                An unexpected error occurred. The details are below.
              </p>
            </div>

            {/* Error message */}
            <div className="p-4 rounded-lg bg-white/5 border border-white/10 text-left">
              <p className="text-[11px] font-mono text-destructive break-all leading-relaxed">
                {this.state.error.message || 'Unknown error'}
              </p>
            </div>

            {/* Actions */}
            <div className="flex gap-3 justify-center">
              <button
                onClick={() => this.setState({ error: null })}
                className="px-4 py-2 rounded-lg bg-white/5 border border-white/10
                           text-sm text-foreground hover:bg-white/10 transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={() => { this.setState({ error: null }); window.location.href = '/'; }}
                className="px-4 py-2 rounded-lg bg-primary/10 border border-primary/30
                           text-sm text-primary hover:bg-primary/20 transition-colors"
              >
                Go Home
              </button>
            </div>

            {/* Disclaimer */}
            <p className="text-[10px] text-muted-foreground/50 font-mono">
              Sambhav may be incorrect. Always verify important decisions independently.
            </p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
      <StickyDisclaimer />
    </ErrorBoundary>
  );
}