import { describe, expect, it } from "vitest";

import { modelLabel, nextBusinessDay, num, pct, signedPct, timeAgo, tone } from "./format";

describe("format helpers", () => {
  it("skips weekends for the next session", () => {
    expect(nextBusinessDay("2026-10-29")).toBe("2026-10-30"); // Thu -> Fri
    expect(nextBusinessDay("2026-10-30")).toBe("2026-11-02"); // Fri -> Mon
  });

  it("maps signals and numbers to a tone", () => {
    expect(tone("BUY")).toBe("up");
    expect(tone("NEGATIVE")).toBe("down");
    expect(tone("HOLD")).toBe("flat");
    expect(tone(-0.3)).toBe("down");
    expect(tone(null)).toBe("flat");
  });

  it("formats numbers and handles missing values", () => {
    expect(pct(0.0891)).toBe("8.9%");
    expect(pct(null)).toBe("–");
    expect(num(Number.NaN)).toBe("–");
    expect(signedPct(0.0617)).toBe("+0.06%");
    expect(signedPct(-1.5)).toBe("-1.50%");
  });

  it("describes headline age", () => {
    const now = Date.parse("2026-10-31T12:00:00Z");
    expect(timeAgo("2026-10-31T09:00:00Z", now)).toBe("3h ago");
    expect(timeAgo("2026-10-29T12:00:00Z", now)).toBe("2 days ago");
    expect(timeAgo(null, now)).toBe("date unknown");
  });

  it("labels models", () => {
    expect(modelLabel("xgboost+news")).toBe("XGBoost + news");
    expect(modelLabel("unknown")).toBe("unknown");
  });
});
