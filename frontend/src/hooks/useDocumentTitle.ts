import { useEffect } from "react";

const BRAND = "SARAS AI";

/** Sets the browser tab title as "<title> · SARAS AI". */
export function useDocumentTitle(title?: string | null): void {
  useEffect(() => {
    document.title = title ? `${title} · ${BRAND}` : BRAND;
  }, [title]);
}
