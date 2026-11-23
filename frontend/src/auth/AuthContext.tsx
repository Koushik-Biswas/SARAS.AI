import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { refreshSession, session } from "../api/client";
import * as api from "../api/endpoints";
import type { LoginIn, RegisterIn, UserOut } from "../api/types";
import { LogoMark } from "../components/Logo";
import { DelayedSpinner } from "../components/Spinner";

type AuthStatus = "loading" | "authenticated" | "anonymous";

interface AuthContextValue {
  status: AuthStatus;
  user: UserOut | null;
  login: (body: LoginIn) => Promise<UserOut>;
  register: (body: RegisterIn) => Promise<UserOut>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/**
 * Session model: access token in memory, refresh token in localStorage.
 * On load the session is restored through POST /auth/refresh (single-flight, so
 * StrictMode double effects do not burn the rotating refresh token twice).
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(null);
  const [status, setStatus] = useState<AuthStatus>(() => (session.hasRefreshToken() ? "loading" : "anonymous"));

  useEffect(() => {
    const unsubscribe = session.subscribe((u) => {
      setUser(u);
      setStatus(u ? "authenticated" : "anonymous");
    });
    if (session.hasRefreshToken() && !session.accessToken) {
      void refreshSession().then((ok) => {
        if (!ok) setStatus(session.accessToken ? "authenticated" : "anonymous");
      });
    }
    return () => {
      unsubscribe();
    };
  }, []);

  const login = useCallback(async (body: LoginIn) => (await api.login(body)).user, []);
  const register = useCallback(async (body: RegisterIn) => (await api.register(body)).user, []);
  const logout = useCallback(async () => {
    await api.logout();
    setUser(null);
    setStatus("anonymous");
  }, []);

  const value = useMemo(() => ({ status, user, login, register, logout }), [status, user, login, register, logout]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

function Splash() {
  return (
    <div className="splash" aria-busy="true">
      <LogoMark size={44} className="splash__mark" />
      <DelayedSpinner loading label="Restoring your session" />
    </div>
  );
}

/** Gate for signed-in routes: waits for session restore, then redirects to /login if anonymous. */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const location = useLocation();
  if (status === "loading") return <Splash />;
  if (status === "anonymous") {
    return <Navigate to="/login" replace state={{ from: `${location.pathname}${location.search}` }} />;
  }
  return <>{children}</>;
}

/** Login/register pages: bounce signed-in users to where they came from. */
export function RedirectIfAuthed({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const location = useLocation();
  if (status === "loading") return <Splash />;
  if (status === "authenticated") {
    const from = (location.state as { from?: string } | null)?.from;
    return <Navigate to={from && from.startsWith("/") ? from : "/"} replace />;
  }
  return <>{children}</>;
}
