"use client";

import katex from "katex";
import { useMemo } from "react";

export function Eq({ tex, id }: { tex: string; id?: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        throwOnError: false,
        displayMode: true,
      }),
    [tex]
  );
  return (
    <div
      id={id}
      className="eq-block overflow-x-auto py-3"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function InlineEq({ tex }: { tex: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        throwOnError: false,
        displayMode: false,
      }),
    [tex]
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
