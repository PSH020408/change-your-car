import assert from "node:assert/strict";
import { after, afterEach, test } from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

// Compile the real transport without adding a browser/test framework dependency.
const out = mkdtempSync(join(tmpdir(), "f1-request-test-"));
execFileSync(process.execPath, ["node_modules/typescript/bin/tsc", "src/lib/request.ts",
  "--target", "ES2022", "--module", "commonjs", "--skipLibCheck", "--outDir", out]);
const { requestJson } = createRequire(import.meta.url)(join(out, "request.js"));
const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
after(() => rmSync(out, { recursive: true, force: true }));

test("successful POST retains the method and body", async () => {
  globalThis.fetch = async (_url, init) => {
    assert.equal(init.method, "POST");
    assert.equal(init.body, '{"setup":1}');
    return Response.json({ delta: -0.101 });
  };
  assert.deepEqual(await requestJson("/api/simulate", { method: "POST", body: '{"setup":1}' }), { delta: -0.101 });
});

test("a server that never responds reaches a finite error", async () => {
  globalThis.fetch = (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
  await assert.rejects(requestJson("/api/baseline", {}, 10), /server did not respond/);
});

test("the deadline also covers a stalled response body", async () => {
  globalThis.fetch = async (_url, { signal }) => ({
    ok: true, headers: new Headers({ "content-type": "application/json" }),
    json: () => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), { once: true });
    }),
  });
  await assert.rejects(requestJson("/api/baseline", {}, 10), /server did not respond/);
});

test("superseded slider requests remain cancellable", async () => {
  globalThis.fetch = (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
  const caller = new AbortController();
  const result = requestJson("/api/simulate", { signal: caller.signal });
  caller.abort();
  await assert.rejects(result, { name: "AbortError" });
});

test("an already-cancelled caller is not silently revived", async () => {
  globalThis.fetch = async (_url, { signal }) => {
    assert.equal(signal.aborted, true);
    throw signal.reason;
  };
  const caller = new AbortController();
  caller.abort();
  await assert.rejects(requestJson("/api/simulate", { signal: caller.signal }), { name: "AbortError" });
});

test("HTTP failures and platform startup HTML produce actionable errors", async () => {
  globalThis.fetch = async () => new Response("unavailable", { status: 503 });
  await assert.rejects(requestJson("/api/baseline"), /HTTP 503/);
  globalThis.fetch = async () => new Response("<html>Starting</html>", { headers: { "content-type": "text/html" } });
  await assert.rejects(requestJson("/api/baseline"), /startup page/);
});
