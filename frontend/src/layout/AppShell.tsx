import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { ChevronRight, LayoutDashboard, LogOut, PanelLeftClose, PanelLeftOpen, Plus } from "lucide-react";
import { useAuth } from "../auth/AuthContext";
import { Avatar, CountBadge, DropdownMenu, IconButton, Logo, MenuItem, MenuLabel, MenuSeparator, Skeleton, ThemeSegmented, ThemeToggleButton, cx } from "../components";
import { useLocalStorage } from "../hooks/useLocalStorage";
import { runGuarded } from "../review/hooks";
import { CrumbProvider, ExamIndexProvider, useCrumbTrail, useExamIndex } from "./ShellContext";

const SIDEBAR_EXAMS = 8;

function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { exams, loading, total } = useExamIndex();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const shown = exams.slice(0, SIDEBAR_EXAMS);

  return (
    <aside className={cx("sidebar", collapsed && "sidebar--collapsed")} aria-label="Primary">
      <div className="sidebar__brand">
        <Link to="/" className="sidebar__logo" aria-label="SARAS AI dashboard">
          <Logo collapsed={collapsed} size={32} />
        </Link>
      </div>

      <nav className="sidebar__nav" aria-label="Main">
        <NavLink to="/" end className={({ isActive }) => cx("nav-item", isActive && "is-active")} title={collapsed ? "Dashboard" : undefined}>
          <LayoutDashboard size={18} aria-hidden="true" />
          <span className="nav-item__label">Dashboard</span>
        </NavLink>

        <div className="sidebar__section">
          <div className="sidebar__section-head">
            <span className="sidebar__section-title">Exams</span>
            <IconButton
              size="sm"
              variant="ghost"
              label="New exam"
              icon={<Plus size={16} aria-hidden="true" />}
              onClick={() => navigate("/?new=1")}
              className="sidebar__new"
            />
          </div>
          {loading ? (
            <div className="sidebar__skeleton">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} height={28} radius={8} />
              ))}
            </div>
          ) : shown.length === 0 ? (
            <p className="sidebar__empty">No exams yet</p>
          ) : (
            <ul className="sidebar__list">
              {shown.map((e) => {
                const active = pathname.startsWith(`/exams/${e.id}`);
                return (
                  <li key={e.id}>
                    <Link
                      to={`/exams/${e.id}`}
                      className={cx("nav-item", "nav-item--exam", active && "is-active")}
                      aria-current={active ? "page" : undefined}
                      title={collapsed ? e.title : undefined}
                    >
                      <span className="nav-item__mono" aria-hidden="true">
                        {e.title.trim().charAt(0).toUpperCase() || "E"}
                      </span>
                      <span className="nav-item__label truncate">{e.title}</span>
                      {!collapsed ? <CountBadge count={e.stats.needs_review} tone="warning" label="answers need review" /> : null}
                    </Link>
                  </li>
                );
              })}
              {total > SIDEBAR_EXAMS && !collapsed ? (
                <li>
                  <Link to="/" className="sidebar__more">
                    View all {total} exams <ChevronRight size={14} aria-hidden="true" />
                  </Link>
                </li>
              ) : null}
            </ul>
          )}
        </div>
      </nav>

      <div className="sidebar__foot">
        <IconButton
          label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          icon={collapsed ? <PanelLeftOpen size={18} aria-hidden="true" /> : <PanelLeftClose size={18} aria-hidden="true" />}
          onClick={onToggle}
          aria-expanded={!collapsed}
        />
      </div>
    </aside>
  );
}

function Breadcrumbs() {
  const trail = useCrumbTrail();
  const items = [{ label: "Dashboard", to: "/" }, ...trail];
  return (
    <nav aria-label="Breadcrumb" className="crumbs">
      <ol>
        {items.map((c, i) => {
          const last = i === items.length - 1;
          return (
            <li key={`${c.label}-${i}`}>
              {!last && c.to ? (
                <Link to={c.to}>{c.label}</Link>
              ) : (
                <span aria-current={last ? "page" : undefined} className={last ? "crumbs__current" : undefined}>
                  {c.label}
                </span>
              )}
              {!last ? <ChevronRight size={14} aria-hidden="true" className="crumbs__sep" /> : null}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function UserMenu() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  if (!user) return null;
  return (
    <DropdownMenu
      label="Account"
      trigger={(props, open) => (
        <button {...props} type="button" className={cx("user-trigger", open && "is-open")} aria-label={`Account menu for ${user.name}`}>
          <Avatar name={user.name} size={30} />
          <span className="user-trigger__name truncate">{user.name}</span>
        </button>
      )}
    >
      {(close) => (
        <>
          <div className="menu__user">
            <Avatar name={user.name} size={38} />
            <div className="menu__user-text">
              <p className="menu__user-name truncate">{user.name}</p>
              <p className="menu__user-email truncate">{user.email}</p>
              {user.role === "admin" ? <p className="menu__user-role">Administrator (read-only marks)</p> : null}
            </div>
          </div>
          <MenuSeparator />
          <MenuLabel>Appearance</MenuLabel>
          <div className="menu__theme">
            <ThemeSegmented />
          </div>
          <MenuSeparator />
          <MenuItem
            danger
            icon={<LogOut size={16} />}
            onSelect={() => {
              close();
              // Signing out drops the page and any unsaved review draft: ask first (runGuarded).
              runGuarded(() => {
                void logout().then(() => navigate("/login", { replace: true }));
              });
            }}
          >
            Sign out
          </MenuItem>
        </>
      )}
    </DropdownMenu>
  );
}

function TopBar() {
  return (
    <header className="topbar">
      <Breadcrumbs />
      <div className="topbar__actions">
        <ThemeToggleButton />
        <span className="topbar__divider" aria-hidden="true" />
        <UserMenu />
      </div>
    </header>
  );
}

/** Signed-in layout: collapsible sidebar, sticky blurred top bar, routed content. */
export function AppShell() {
  const [collapsed, setCollapsed] = useLocalStorage("saras-sidebar-collapsed", false);
  const { pathname } = useLocation();
  return (
    <ExamIndexProvider>
      <CrumbProvider>
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <div className={cx("shell", collapsed && "shell--collapsed")}>
          <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
          <div className="shell__main">
            <TopBar />
            <main id="main" className="content" tabIndex={-1}>
              <div key={pathname.split("/").slice(0, 3).join("/")} className="page-enter">
                <Outlet />
              </div>
            </main>
          </div>
        </div>
      </CrumbProvider>
    </ExamIndexProvider>
  );
}
