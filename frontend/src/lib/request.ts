/** Allow a sleeping demo instance to start, but never leave the HUD waiting forever. */
export async function requestJson<T>(url: string, init: RequestInit = {}, timeoutMs = 90_000): Promise<T> {
  const controller = new AbortController();
  const callerSignal = init.signal;
  const cancel = () => controller.abort(callerSignal?.reason);
  let timedOut = false;
  if (callerSignal?.aborted) cancel();
  else callerSignal?.addEventListener("abort", cancel, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) throw new Error(`Request failed (HTTP ${res.status}). Please retry shortly.`);
    if (!res.headers.get("content-type")?.includes("application/json")) {
      throw new Error("The server returned a startup page instead of data. Please retry shortly.");
    }
    // Keep the deadline active while the response body is being received too.
    return await res.json() as T;
  } catch (error) {
    if (timedOut) throw new Error(`The server did not respond within ${Math.ceil(timeoutMs / 1000)} seconds. It may be starting or temporarily unavailable. Please retry.`);
    throw error;
  } finally {
    clearTimeout(timer);
    callerSignal?.removeEventListener("abort", cancel);
  }
}
