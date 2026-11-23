import { ButtonLink, EmptyState } from "../components";
import { useDocumentTitle } from "../hooks/useDocumentTitle";

export default function NotFoundPage() {
  useDocumentTitle("Page not found");
  return (
    <div className="page">
      <div className="card">
        <EmptyState
          title="Page not found"
          description="The page you were looking for doesn't exist or has moved."
          action={
            <ButtonLink to="/" variant="primary">
              Back to dashboard
            </ButtonLink>
          }
        />
      </div>
    </div>
  );
}
