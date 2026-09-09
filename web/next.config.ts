import path from "node:path";

import type { NextConfig } from "next";

/*
  Static export, deliberately.

  One build has to satisfy two very different hosts: Vercel, and a Starlette
  `StaticFiles` mount inside `hybrid-rag serve`. An export is the only output
  both can serve, and it costs nothing here - the console is entirely
  client-side, because the only data source is an API on the viewer's own
  machine that a build server could never reach.
*/
const nextConfig: NextConfig = {
  /*
    Pin the workspace root. Turbopack walks upwards looking for a lockfile and
    finds a stray `package-lock.json` in `Kevin codes/`, outside this git repo -
    which makes the inferred root depend on what happens to be sitting in a
    parent directory. Anchoring it here keeps the build the same everywhere,
    including on Vercel.
  */
  turbopack: { root: path.resolve(process.cwd()) },
  output: "export",
  // `next/image` optimisation needs a server. There is none in an export.
  images: { unoptimized: true },
  // Emit `out/index.html` rather than `out/index/index.html`, so the static
  // mount resolves `/` without directory-index rules.
  trailingSlash: false,
};

export default nextConfig;
