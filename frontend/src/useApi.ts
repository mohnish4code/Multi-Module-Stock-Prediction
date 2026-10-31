import { useCallback, useEffect, useRef, useState } from "react";

interface State<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Runs `load` whenever `key` changes (null = don't run). Stale responses are aborted. */
export function useApi<T>(load: (signal: AbortSignal) => Promise<T>, key: string | null) {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: key != null });
  const [version, setVersion] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (key == null) return;
    const controller = new AbortController();
    setState((s) => ({ data: s.data, error: null, loading: true }));
    loadRef
      .current(controller.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((err: Error) => {
        if (err.name !== "AbortError") setState({ data: null, error: err.message, loading: false });
      });
    return () => controller.abort();
  }, [key, version]);

  const reload = useCallback(() => setVersion((v) => v + 1), []);
  return { ...state, reload };
}
