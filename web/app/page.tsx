import { Console } from "@/components/console";

/*
  Full-height shell. The header and the composer are fixed chrome, and the
  transcript scrolls between them. `100dvh` rather than `100vh` so the composer
  is not hidden behind a mobile browser's collapsing toolbar.

  The header lives inside `Console` because the lockup doubles as "home" and
  therefore has to reach the transcript state.
*/
export default function Page() {
  return (
    <div className="flex h-[100dvh] flex-col">
      <Console />
    </div>
  );
}
