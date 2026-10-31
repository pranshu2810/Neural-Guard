import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  ScanSearch,
  ShieldCheck,
  History,
  Settings,
  Shield,
} from "lucide-react";

const navItems = [
  { path: "/", label: "Overview", icon: LayoutDashboard },
  { path: "/scan", label: "Scan Model", icon: ScanSearch },
  { path: "/baselines", label: "Trusted Models", icon: ShieldCheck },
  { path: "/history", label: "Scan History", icon: History },
  { path: "/settings", label: "Settings", icon: Settings },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-64 border-r border-white/5 bg-[var(--color-navy-900)] flex flex-col shrink-0">
        {/* Logo */}
        <div className="p-6 flex items-center gap-3 border-b border-white/5">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight">
              Neural Guard
            </h1>
            <p className="text-[10px] text-[var(--color-slate-500)] uppercase tracking-widest">
              AI Model Security
            </p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === "/"}
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-white/5">
          <p className="text-[10px] text-[var(--color-slate-600)] text-center">
            v0.1.0 · Protecting AI Models
          </p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-6 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
