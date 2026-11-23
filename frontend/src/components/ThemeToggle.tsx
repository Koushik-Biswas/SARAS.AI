import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemeMode } from "../theme/ThemeProvider";
import { IconButton } from "./Button";
import { Segmented } from "./Segmented";

const NEXT: Record<ThemeMode, ThemeMode> = { light: "dark", dark: "system", system: "light" };
const NAMES: Record<ThemeMode, string> = { light: "Light", dark: "Dark", system: "System" };

/** Top-bar icon button that cycles light -> dark -> system. */
export function ThemeToggleButton() {
  const { mode, resolved, cycle } = useTheme();
  const icon = mode === "system" ? <Monitor size={18} aria-hidden="true" /> : resolved === "dark" ? <Moon size={18} aria-hidden="true" /> : <Sun size={18} aria-hidden="true" />;
  const label = `Theme: ${NAMES[mode]}${mode === "system" ? ` (${resolved})` : ""}. Switch to ${NAMES[NEXT[mode]]}`;
  return <IconButton label={label} icon={icon} onClick={cycle} className="theme-toggle" />;
}

/** Light / Dark / System segmented control (user menu, login corner). */
export function ThemeSegmented({ size = "sm" }: { size?: "sm" | "md" }) {
  const { mode, setMode } = useTheme();
  return (
    <Segmented<ThemeMode>
      label="Colour theme"
      size={size}
      value={mode}
      onChange={setMode}
      options={[
        { value: "light", label: "Light", icon: <Sun size={14} aria-hidden="true" /> },
        { value: "dark", label: "Dark", icon: <Moon size={14} aria-hidden="true" /> },
        { value: "system", label: "System", icon: <Monitor size={14} aria-hidden="true" /> },
      ]}
    />
  );
}
