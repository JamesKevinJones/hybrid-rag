import type { Metadata } from "next";
import localFont from "next/font/local";

import "./globals.css";

/*
  Monaspace, self-hosted.

  It is not on Google Fonts, so these are `next/font/local` over woff2 copied
  from @fontsource at build time. Self-hosting also means no third-party
  request from a page whose entire premise is that it does not phone home.

  Three of the five family members, each doing the job it was cut for:

    Krypton  mechanical, squared    -> display. Reads as engineered.
    Argon    humanist               -> body. The most readable of the family
                                      at paragraph length.
    Neon     neo-grotesque          -> labels, badges, numbers. The terminal
                                      voice, and it matches the TUI.

  The whole superfamily is monospaced, which is the point: this is a console,
  and every glyph advancing the same width is what makes a column of latencies
  and hashes scan. It also costs width - mono runs ~15% wider than a
  proportional face at the same size, so the display clamp is tuned lower than
  it would be for Archivo.
*/

const krypton = localFont({
  variable: "--font-display",
  display: "swap",
  src: [
    { path: "./fonts/monaspace-krypton-latin-700-normal.woff2", weight: "700", style: "normal" },
    { path: "./fonts/monaspace-krypton-latin-800-normal.woff2", weight: "800", style: "normal" },
  ],
});

const argon = localFont({
  variable: "--font-body",
  display: "swap",
  src: [
    { path: "./fonts/monaspace-argon-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/monaspace-argon-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/monaspace-argon-latin-700-normal.woff2", weight: "700", style: "normal" },
  ],
});

const neon = localFont({
  variable: "--font-mono",
  display: "swap",
  src: [
    { path: "./fonts/monaspace-neon-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/monaspace-neon-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/monaspace-neon-latin-600-normal.woff2", weight: "600", style: "normal" },
  ],
});

export const metadata: Metadata = {
  title: "hybrid-rag console",
  description:
    "A local-first RAG system that treats cloud tokens as a budgeted resource, " +
    "and can prove afterwards how it spent them.",
};

/*
  Theme is set before first paint, from the DOM rather than React state.

  Doing it in an effect means the browser paints the default theme first and
  then swaps, which is a visible flash on every load. Doing it in state means
  the server-rendered markup disagrees with the client, so `suppressHydration
  Warning` below is deliberate rather than a papered-over bug: the script is
  supposed to change the markup before React ever sees it.
*/
const THEME_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem("hybrid-rag-theme");
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var theme = stored || (prefersDark ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
  } catch (e) {
    /* A browser with storage blocked still gets a readable page. */
  }
})();
`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className={`${krypton.variable} ${argon.variable} ${neon.variable}`}>
        <a
          href="#composer"
          className="brut-sm sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:bg-acid focus:px-4 focus:py-2 focus:text-ink"
        >
          Skip to the prompt
        </a>
        {children}
      </body>
    </html>
  );
}
