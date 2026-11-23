import { useEffect, useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { CircleCheck, Eye, EyeOff, Lock, Mail, ScanText, ShieldCheck, Sparkles, User } from "lucide-react";
import { ApiError, errorMessage } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Badge, Banner, Button, ConfidenceMeter, Field, IconButton, Input, Logo, ThemeSegmented } from "../components";
import { useDocumentTitle } from "../hooks/useDocumentTitle";

type Mode = "login" | "register";

const VALUES = [
  { icon: ScanText, title: "Reads handwriting", text: "Scanned answer sheets are cleaned, read and split by question automatically." },
  { icon: Sparkles, title: "Grades against your rubric", text: "Every mark is traced to the key points it matched, with a confidence score." },
  { icon: ShieldCheck, title: "You stay in control", text: "Low-confidence answers go to review, and every change is audit-logged." },
];

function PreviewCard() {
  return (
    <div className="auth-preview" aria-hidden="true">
      <div className="auth-preview__head">
        <span className="auth-preview__q">Q3 · Short answer</span>
        <Badge tone="success" size="sm" icon={<CircleCheck size={12} />}>
          Auto-accepted
        </Badge>
      </div>
      <p className="auth-preview__text">"Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen."</p>
      <ul className="auth-preview__points">
        <li className="is-hit">
          <CircleCheck size={14} /> Light energy to chemical energy
        </li>
        <li className="is-hit">
          <CircleCheck size={14} /> Glucose is produced
        </li>
        <li className="is-hit">
          <CircleCheck size={14} /> Oxygen is released
        </li>
      </ul>
      <div className="auth-preview__foot">
        <span className="auth-preview__marks num">
          4.5 <span>/ 5</span>
        </span>
        <ConfidenceMeter value={0.91} compact />
      </div>
    </div>
  );
}

function useCountdown(seconds: number | null): number {
  const [left, setLeft] = useState(0);
  useEffect(() => {
    if (!seconds) {
      setLeft(0);
      return;
    }
    const end = Date.now() + seconds * 1000;
    setLeft(seconds);
    const t = window.setInterval(() => {
      const s = Math.max(0, Math.ceil((end - Date.now()) / 1000));
      setLeft(s);
      if (s === 0) window.clearInterval(t);
    }, 1000);
    return () => window.clearInterval(t);
  }, [seconds]);
  return left;
}

function formatWait(s: number): string {
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")} min` : `${r} s`;
}

export default function LoginPage({ mode }: { mode: Mode }) {
  const isLogin = mode === "login";
  useDocumentTitle(isLogin ? "Sign in" : "Create account");
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [lockSeconds, setLockSeconds] = useState<number | null>(null);
  const lockLeft = useCountdown(lockSeconds);

  useEffect(() => {
    setError(null);
    setFieldErrors({});
  }, [mode]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy || lockLeft > 0) return;
    setBusy(true);
    setError(null);
    setFieldErrors({});
    try {
      if (isLogin) await login({ email: email.trim(), password });
      else await register({ name: name.trim(), email: email.trim(), password });
      const from = (location.state as { from?: string } | null)?.from;
      navigate(from && from.startsWith("/") ? from : "/", { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setLockSeconds(err.retryAfterSeconds ?? 300);
        setError(null);
      } else if (err instanceof ApiError && err.status === 401) {
        // One generic message: never reveal whether the email exists.
        setError("Email or password is incorrect.");
      } else if (err instanceof ApiError && err.status === 400 && !isLogin) {
        setFieldErrors(err.fieldErrors);
        setError(Object.keys(err.fieldErrors).length ? null : err.message);
      } else if (err instanceof ApiError && err.status === 400) {
        setError("Email or password is incorrect.");
      } else if (err instanceof ApiError && err.status === 409) {
        setFieldErrors({ email: "An account with this email already exists." });
      } else {
        setError(errorMessage(err));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <section className="auth__brand" aria-label="About SARAS AI">
        <div className="auth__mesh" aria-hidden="true" />
        <div className="auth__brand-inner">
          <Logo size={40} tagline />
          <div className="auth__pitch">
            <h1 className="auth__headline">Mark a stack of answer sheets in minutes, not evenings.</h1>
            <p className="auth__lead">
              SARAS AI reads scanned scripts, grades each answer against your rubric and shows you exactly why, so you only review what needs a
              human eye.
            </p>
          </div>
          <ul className="auth__values">
            {VALUES.map(({ icon: Icon, title, text }) => (
              <li key={title}>
                <span className="auth__value-icon" aria-hidden="true">
                  <Icon size={18} />
                </span>
                <div>
                  <p className="auth__value-title">{title}</p>
                  <p className="auth__value-text">{text}</p>
                </div>
              </li>
            ))}
          </ul>
          <PreviewCard />
        </div>
      </section>

      <section className="auth__panel">
        <div className="auth__corner">
          <ThemeSegmented />
        </div>
        <div className="auth__card">
          <div className="auth__card-head">
            <h2>{isLogin ? "Welcome back" : "Create your account"}</h2>
            <p className="muted">{isLogin ? "Sign in to continue grading." : "Set up a teacher account in under a minute."}</p>
          </div>

          {lockLeft > 0 ? (
            <Banner tone="warning" title="Too many sign-in attempts">
              For your security, sign-in is paused. Try again in <span className="num">{formatWait(lockLeft)}</span>.
            </Banner>
          ) : null}
          {error ? <Banner tone="danger" title={error} /> : null}

          <form className="auth__form" onSubmit={onSubmit} noValidate>
            {!isLogin ? (
              <Field label="Full name" error={fieldErrors.name} required>
                <Input
                  iconLeft={<User size={16} />}
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={120}
                  inputSize="lg"
                  placeholder="e.g. Priya Sharma"
                />
              </Field>
            ) : null}
            <Field label="Email" error={fieldErrors.email} required>
              <Input
                iconLeft={<Mail size={16} />}
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                inputSize="lg"
                placeholder="you@school.edu"
                autoFocus
              />
            </Field>
            <Field label="Password" error={fieldErrors.password} hint={!isLogin ? "At least 8 characters." : undefined} required>
              <Input
                iconLeft={<Lock size={16} />}
                type={showPw ? "text" : "password"}
                autoComplete={isLogin ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                inputSize="lg"
                endAdornment={
                  <IconButton
                    size="sm"
                    label={showPw ? "Hide password" : "Show password"}
                    icon={showPw ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
                    onClick={() => setShowPw((s) => !s)}
                    aria-pressed={showPw}
                  />
                }
              />
            </Field>
            <Button
              type="submit"
              variant="primary"
              size="lg"
              block
              loading={busy}
              disabled={lockLeft > 0 || !email || !password || (!isLogin && (!name.trim() || password.length < 8))}
            >
              {isLogin ? "Sign in" : "Create account"}
            </Button>
          </form>

          <p className="auth__switch">
            {isLogin ? (
              <>
                New to SARAS AI? <Link to="/register" state={location.state}>Create an account</Link>
              </>
            ) : (
              <>
                Already have an account? <Link to="/login" state={location.state}>Sign in</Link>
              </>
            )}
          </p>
        </div>
        <p className="auth__foot">Smart Assessment and Recognition Automation System</p>
      </section>
    </div>
  );
}
