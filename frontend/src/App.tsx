import { Navigate, Route, RouterProvider, createBrowserRouter, createRoutesFromElements, useParams } from "react-router-dom";
import { AuthProvider, RedirectIfAuthed, RequireAuth } from "./auth/AuthContext";
import { ToastProvider } from "./components";
import { AppShell } from "./layout/AppShell";
import AnalyticsPage from "./pages/AnalyticsPage";
import ApprovalPage from "./pages/ApprovalPage";
import DashboardPage from "./pages/DashboardPage";
import ExamLayout, { ExamIndexRedirect } from "./pages/ExamLayout";
import LoginPage from "./pages/LoginPage";
import NotFoundPage from "./pages/NotFoundPage";
import RegionEditorPage from "./pages/RegionEditorPage";
import ResultsPage from "./pages/ResultsPage";
import ReviewPage from "./pages/ReviewPage";
import RubricPage from "./pages/RubricPage";
import StatusPage from "./pages/StatusPage";
import UploadPage from "./pages/UploadPage";
import { ThemeProvider } from "./theme/ThemeProvider";

/** Passes URL params to ReviewPage as props (submissionId optional). */
function ReviewRoute() {
  const { examId = "", submissionId } = useParams();
  return <ReviewPage examId={examId} submissionId={submissionId} />;
}

/** Passes URL params to RegionEditorPage as props. */
function RegionsRoute() {
  const { examId = "", submissionId = "" } = useParams();
  return <RegionEditorPage key={submissionId} examId={examId} submissionId={submissionId} />;
}

// A data router (not <BrowserRouter>): screens with unsaved work can then hold every
// navigation with useBlocker, including the browser's Back/Forward buttons.
const router = createBrowserRouter(
  createRoutesFromElements(
    <>
      <Route
        path="/login"
        element={
          <RedirectIfAuthed>
            <LoginPage mode="login" />
          </RedirectIfAuthed>
        }
      />
      <Route
        path="/register"
        element={
          <RedirectIfAuthed>
            <LoginPage mode="register" />
          </RedirectIfAuthed>
        }
      />
      <Route
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route index element={<DashboardPage />} />
        <Route path="exams/:examId" element={<ExamLayout />}>
          <Route index element={<ExamIndexRedirect />} />
          <Route path="rubric" element={<RubricPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="status" element={<StatusPage />} />
          <Route path="review" element={<ReviewRoute />} />
          <Route path="review/:submissionId" element={<ReviewRoute />} />
          <Route path="regions/:submissionId" element={<RegionsRoute />} />
          <Route path="results" element={<ResultsPage />} />
          <Route path="approval" element={<ApprovalPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="*" element={<Navigate to=".." replace />} />
        </Route>
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </>,
  ),
);

export default function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <AuthProvider>
          <RouterProvider router={router} />
        </AuthProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}
