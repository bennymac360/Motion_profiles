#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Profile Designer
(Multiprofile • Move/Wait steps • Waypoint speed cap • Optional approach accel limit • Per-step jerk)
----------------------------------------------------------------------------------------------------
Single- or multi-segment, physically-consistent 1D motion profiles.

Global limits: v_max, a_max, d_max, j_max

Steps table (≤10 rows). Each row is a **Step**:
  • Step Type = Move:
      - Target x [m] ............... position of the waypoint
      - Target v (speed) [m/s] ..... desired **speed at the waypoint** (magnitude)
          * sign inferred from direction (Δx)
          * also acts as a **local peak-speed cap**: v_peak ≤ min(v_max, |Target v|)
      - Target a (opt, signed) ..... only tightens the **active** accel limit:
          • if speeding up (|vf|>|v0|) and a_pref ≥ 0  → a_max_seg = min(|a_pref|, a_max)
          • if slowing  (|vf|<|v0|) and a_pref ≤ 0  → d_max_seg = min(|a_pref|, d_max)
        (Global caps remain hard; per-row only tightens.)
      - Jerk j (opt) [m/s³] ........ per-step jerk cap:
          • J_step = min(|j|, j_max_global) if j given and j>0
          • leave blank to use global j_max
          • j ≤ 0 forces **no jerk limit** for the step (trapezoid/triangle)
      - Automatic stop rules:
          • if the **next step is WAIT**, this Move ends at **vf=0**
          • if the **next MOVE reverses direction**, we force **vf=0** here
  • Step Type = Wait:
      - Wait t [s] ................. dwell at current x with v=0 for t seconds
      - Requires v=0 at the start (planner forces the stop on the previous Move)

If j_max > 0 → jerk-limited S-curve per segment (7-seg; collapses to triangular).
If j_max ≤ 0 → constant-accel trapezoid/triangle.

Properties:
  - Exact position & velocity continuity across segments
  - a(t)=0 and jerk(t)=0 at every boundary
  - Clear feasibility errors if a segment cannot meet the constraints/distance

Author: ChatGPT
License: MIT
"""
import math
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX setup
# ──────────────────────────────────────────────────────────────────────────────
def _detect_latex():
    import shutil, subprocess
    for exe in ("latex", "pdflatex", "xelatex", "lualatex"):
        if shutil.which(exe):
            try:
                out = subprocess.run([exe, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
                if out.returncode == 0:
                    return True
            except Exception:
                pass
    return False

HAS_LATEX = _detect_latex()
matplotlib.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.size": 11,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "text.usetex": HAS_LATEX,
})

COLORS = {
    "pos":  "tab:blue",
    "vel":  "tab:red",
    "acc":  "tab:green",
    "jerk": "tab:orange",
}

def _sgn(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


# ──────────────────────────────────────────────────────────────────────────────
# Trapezoid planner (supports per-segment waypoint speed cap)
# ──────────────────────────────────────────────────────────────────────────────
def plan_trapezoid(p0, pf, v0, vf, vmax_global, a_max, d_max, v_cap=None):
    """
    Constant-accel trapezoid/triangle solving (x0,v0)→(xf,vf).
    v_cap (optional): local *peak-speed* cap; effective_cap = min(vmax_global, |v_cap|).
    Peak never below boundary speeds in along-direction frame.
    """
    if a_max <= 0 or d_max <= 0:
        raise ValueError("Acceleration and deceleration must be positive.")
    if vmax_global < 0:
        raise ValueError("Max velocity must be non-negative.")

    if p0 == pf and abs(v0 - vf) < 1e-12:
        return {
            "type": "idle",
            "sgn": 1.0,
            "t": np.array([0.0, 0.0, 0.0, 0.0]),
            "durations": (0.0, 0.0, 0.0),
            "v_profile": (v0, v0, vf),
            "a": (0.0, 0.0, 0.0),
            "summary": "No motion: x0 == xf and v0 == vf.",
            "T": 0.0,
            "v0s": 0.0, "vfs": 0.0, "vpk": 0.0, "vmax_eff": 0.0,
            "da": 0.0, "dd": 0.0, "tc": 0.0,
        }

    sgn = _sgn(pf - p0)
    L = abs(pf - p0)

    # Speeds along direction (non-negative)
    v0s = max(0.0, sgn * v0)
    vfs = max(0.0, sgn * vf)

    vmax_cap = vmax_global if v_cap is None else min(vmax_global, abs(v_cap))
    vmax_eff = max(vmax_cap, v0s, vfs)

    a = float(abs(a_max))
    d = float(abs(d_max))

    # Times/distances
    ta = max(0.0, (vmax_eff - v0s) / a)
    da = (v0s + vmax_eff) * 0.5 * ta

    td = max(0.0, (vmax_eff - vfs) / d)
    dd = (vfs + vmax_eff) * 0.5 * td

    Lc = L - da - dd

    if Lc >= -1e-12:
        tc = max(0.0, Lc / max(vmax_eff, 1e-12))
        profile_type = "trapezoid"
        vpk = vmax_eff
    else:
        # Triangular: solve v_peak
        num = 2.0 * L * a * d + d * (v0s ** 2) + a * (vfs ** 2)
        den = a + d
        vpk2 = num / den
        vpk = math.sqrt(max(vpk2, 0.0))
        if vpk < v0s - 1e-9 or vpk < vfs - 1e-9:
            raise ValueError("Infeasible: v_peak < boundary speed.")
        ta = (vpk - v0s) / a if a > 0 else 0.0
        td = (vpk - vfs) / d if d > 0 else 0.0
        tc = 0.0
        profile_type = "triangle"
        da = (v0s + vpk) * 0.5 * ta
        dd = (vfs + vpk) * 0.5 * td

    T = ta + tc + td
    a_acc = sgn * a
    a_dec = -sgn * d

    summary = "\n".join([
        f"Profile: {profile_type.upper()} (no jerk limit)",
        f"Distance |L| = {L:.6g} m, sgn = {int(sgn):+d}",
        f"t_acc={ta:.6g} s, t_cruise={tc:.6g} s, t_dec={td:.6g} s, total T={T:.6g} s",
        f"Speeds: v0={v0:.6g} m/s, v_peak={sgn*vpk:.6g} m/s, v_final={vf:.6g} m/s",
        f"Waypoint cap = min(v_max, target v) = {vmax_cap:.6g} m/s",
    ])

    return {
        "type": profile_type,
        "sgn": sgn,
        "t": np.array([0.0, ta, ta + tc, T]),
        "durations": (ta, tc, td),
        "v_profile": (v0, sgn * vpk, vf),
        "a": (a_acc, 0.0, a_dec),
        "summary": summary,
        "L": L, "T": T, "v0s": v0s, "vfs": vfs, "vpk": vpk, "vmax_eff": vmax_eff,
        "da": da, "dd": dd, "tc": tc,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Jerk-limited S-curve planner (supports per-segment waypoint speed cap)
# ──────────────────────────────────────────────────────────────────────────────
def _Q(tj: float, tc: float) -> float:
    # distance contribution during a jerk-limited monotone phase
    return tj * tj + 1.5 * tj * tc + 0.5 * tc * tc

def _phase_jlimited(v_in: float, v_out: float, a_lim: float, J: float):
    """
    Monotone speed change with jerk limit J and accel limit a_lim.
    Returns (tj, t_const, t_total, s, triangular)
    """
    inc = v_out >= v_in
    dv = abs(v_out - v_in)
    tj_lim = a_lim / J
    dv_to_hit_alim = a_lim * tj_lim  # = a_lim^2 / J

    if dv + 1e-12 >= dv_to_hit_alim:
        tj = tj_lim
        t_const = dv / a_lim - tj
        t_total = t_const + 2.0 * tj
        if inc:
            s = v_in * t_total + a_lim * _Q(tj, t_const)
        else:
            s = v_in * t_total - a_lim * _Q(tj, t_const)
        return tj, t_const, t_total, s, False
    else:
        tj = math.sqrt(dv / J)
        t_const = 0.0
        t_total = 2.0 * tj
        if inc:
            s = 2.0 * v_in * tj + dv * tj
        else:
            s = 2.0 * v_in * tj - dv * tj
        return tj, t_const, t_total, s, True

def _s_total_for_vpk(v0s, vfs, vpk, a, d, J):
    return _phase_jlimited(v0s, vpk, a, J)[3] + _phase_jlimited(vpk, vfs, d, J)[3]

def _find_vpk(L, v0s, vfs, vmax_eff, a, d, J):
    v_low = max(v0s, vfs)
    s_min = _s_total_for_vpk(v0s, vfs, v_low, a, d, J)
    if s_min - L > 1e-9:
        return None, s_min, 'infeasible'
    s_high = _s_total_for_vpk(v0s, vfs, vmax_eff, a, d, J)
    if s_high <= L + 1e-12:
        return vmax_eff, s_high, 'cruise'
    lo, hi = v_low, vmax_eff
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        sm = _s_total_for_vpk(v0s, vfs, mid, a, d, J)
        if sm < L:
            lo = mid
        else:
            hi = mid
    vpk = 0.5 * (lo + hi)
    return vpk, _s_total_for_vpk(v0s, vfs, vpk, a, d, J), 'triangle'

def plan_profile(p0, pf, v0, vf, vmax_global, a_max, d_max, j_max, v_cap=None):
    """
    If j_max > 0 ⇒ jerk-limited S-curve; else fallback to trapezoid.
    v_cap (optional): local *peak-speed* cap ⇒ v_peak ≤ min(v_max, |v_cap|).
    """
    if j_max is None or not math.isfinite(j_max) or j_max <= 0:
        return plan_trapezoid(p0, pf, v0, vf, vmax_global, a_max, d_max, v_cap=v_cap)

    if a_max <= 0 or d_max <= 0:
        raise ValueError("Acceleration limits must be positive.")
    if vmax_global < 0:
        raise ValueError("v_max must be non-negative.")

    sgn = _sgn(pf - p0)
    L = abs(pf - p0)
    v0s = max(0.0, sgn * v0)
    vfs = max(0.0, sgn * vf)

    a = float(abs(a_max))
    d = float(abs(d_max))
    J = float(abs(j_max))

    if L == 0.0 and abs(v0 - vf) <= 1e-12:
        return {
            "type": "idle",
            "sgn": sgn,
            "durations": (0.0, 0.0, 0.0),
            "T": 0.0,
            "v_profile": (v0, v0, vf),
            "jerk_segments": [],
            "summary": "No motion: x0 == xf and v0 == vf.",
        }

    vmax_cap = vmax_global if v_cap is None else min(vmax_global, abs(v_cap))
    vmax_eff = max(vmax_cap, v0s, vfs)

    vpk, s_at, kind = _find_vpk(L, v0s, vfs, vmax_eff, a, d, J)
    if kind == 'infeasible':
        raise ValueError(f"Infeasible monotonic motion: minimal distance {s_at:.6g} m > |L|={L:.6g} m.")

    tj_a, ta_c, ta_tot, s_acc, _ = _phase_jlimited(v0s, vpk, a, J)
    tj_d, td_c, td_tot, s_dec, _ = _phase_jlimited(vpk, vfs, d, J)
    tc = (L - s_acc - s_dec) / max(vpk, 1e-12) if kind == 'cruise' else 0.0

    ta = ta_tot
    td = td_tot
    T = ta + tc + td

    segs = []
    if tj_a > 1e-12: segs.append(("acc_j+", +1, tj_a))
    if ta_c > 1e-12: segs.append(("acc_const", 0, ta_c))
    if tj_a > 1e-12: segs.append(("acc_j-", -1, tj_a))
    if tc  > 1e-12: segs.append(("cruise", 0, tc))
    if tj_d > 1e-12: segs.append(("dec_j-", -1, tj_d))
    if td_c > 1e-12: segs.append(("dec_const", 0, td_c))
    if tj_d > 1e-12: segs.append(("dec_j+", +1, tj_d))

    world_segs = [{"name": name, "j": sgn * jsign * J, "dt": dt} for (name, jsign, dt) in segs]
    vpk_signed = sgn * vpk

    summary_lines = [
        f"Profile: jerk-limited {'S-curve+cruise' if tc>1e-12 else 'S-curve (triangular)'}",
        f"|L|={L:.6g} m, sgn={int(sgn):+d}",
        f"t_acc={ta:.6g} s, t_cruise={tc:.6g} s, t_dec={td:.6g} s, T={T:.6g} s",
        f"Speeds: v0={v0:.6g} → v_peak={vpk_signed:.6g} → vf={vf:.6g}",
        f"Waypoint cap = min(v_max, target v) = {vmax_cap:.6g} m/s",
        f"Jerk cap used J={J:.6g} m/s³",
    ]

    return {
        "type": "s-curve" if tc > 1e-12 else "s-curve-tri",
        "sgn": sgn,
        "durations": (ta, tc, td),
        "T": T,
        "v_profile": (v0, vpk_signed, vf),
        "jerk_segments": world_segs,
        "summary": "\n".join(summary_lines),
        "params": {"L": L, "v0s": v0s, "vfs": vfs, "vpk": vpk, "a": a, "d": d, "J": J},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-profile planner  (Move/Wait + per-step jerk)
# ──────────────────────────────────────────────────────────────────────────────
def plan_multiprofile(x0, v0, steps, vmax, amax, dmax, jmax):
    """
    steps: list of dicts. Each dict is either
      {'type':'move','xf':float,'v_speed':float,'a_pref':float|None,'j_pref':float|None}
      {'type':'wait','dt':float}

    Rules:
      - v_speed is the desired speed **at** the waypoint (magnitude).
        Sign inferred from segment direction; also used as local peak-speed cap.
      - If the next step is WAIT, the current MOVE is forced to end with vf=0.
      - If the next MOVE reverses direction, current MOVE forced to end with vf=0.
      - WAIT requires v=0 at its start (enforced by the look-ahead rules).
      - Per-step jerk: J_step = min(|j_pref|, jmax) if j_pref>0; blank→jmax; j_pref≤0 → no-jerk trapezoid.
    """
    if not steps:
        raise ValueError("No steps provided.")

    # Preprocess: infer signs, caps, and enforce stops before waits / reversals
    processed = []
    x_cursor = x0
    prev_dir = None

    # Helper: next move direction seen after any waits
    def _next_move_dir(idx, from_x, fallback_dir):
        for j in range(idx + 1, len(steps)):
            st = steps[j]
            if st["type"].strip().lower() == "move":
                dx = st["xf"] - from_x
                return _sgn(dx) if abs(dx) > 0 else (fallback_dir if fallback_dir is not None else +1.0)
        return fallback_dir if fallback_dir is not None else +1.0

    for i, st in enumerate(steps):
        stype = st["type"].strip().lower()
        if stype == "move":
            xf = float(st["xf"])
            vmag = abs(float(st["v_speed"]))
            dx = xf - x_cursor
            dir_i = _sgn(dx) if abs(dx) > 0 else (prev_dir if prev_dir is not None else +1.0)

            # Signed waypoint speed & local cap
            vf_signed = dir_i * vmag
            vcap = vmag

            # look-ahead: stop before wait or reversal
            next_is_wait = (i + 1 < len(steps) and steps[i + 1]["type"].strip().lower() == "wait")
            dir_next_move = _next_move_dir(i, xf, dir_i)
            if next_is_wait or (dir_next_move != dir_i):
                vf_signed = 0.0

            processed.append({
                "type": "move",
                "xf": xf,
                "vf": vf_signed,
                "vcap": vcap,
                "a_pref": st.get("a_pref"),
                "j_pref": st.get("j_pref"),
            })
            x_cursor = xf
            prev_dir = dir_i

        elif stype == "wait":
            dt = float(st["dt"])
            if dt < 0:
                raise ValueError("Wait time must be non-negative.")
            processed.append({"type": "wait", "dt": dt})
        else:
            raise ValueError("Step type must be 'Move' or 'Wait'.")

    # If the very first step is WAIT, require initial v0==0.
    if processed and processed[0]["type"] == "wait" and abs(v0) > 1e-9:
        raise ValueError("First step is WAIT but initial velocity v0 ≠ 0. Insert a stopping Move or set v0=0.")

    # Plan each step
    plans = []
    x_curr, v_curr = x0, v0
    for st in processed:
        if st["type"] == "move":
            a_use, d_use = amax, dmax
            a_pref = st.get("a_pref")
            vf_signed = st["vf"]
            # Tighten accel/dec depending on speed-up/slow-down
            if a_pref is not None and math.isfinite(a_pref):
                if abs(vf_signed) > abs(v_curr) + 1e-12:      # speeding up
                    if a_pref >= 0:
                        a_use = min(abs(a_pref), amax)
                elif abs(vf_signed) < abs(v_curr) - 1e-12:   # slowing
                    if a_pref <= 0:
                        d_use = min(abs(a_pref), dmax)

            # Per-step jerk choice
            j_pref = st.get("j_pref")
            if j_pref is None:
                j_use = jmax
            else:
                j_pref = float(j_pref)
                if j_pref <= 0:
                    j_use = 0.0  # force trapezoid
                else:
                    j_use = min(abs(j_pref), jmax)

            plan = plan_profile(x_curr, st["xf"], v_curr, vf_signed,
                                vmax_global=vmax, a_max=a_use, d_max=d_use, j_max=j_use,
                                v_cap=st["vcap"])
            plans.append(plan)
            x_curr, v_curr = st["xf"], vf_signed

        else:  # wait
            if abs(v_curr) > 1e-9:
                raise ValueError("Wait step requires the axis to be stopped (v=0) at its start. "
                                 "The previous Move is automatically forced to stop; check your inputs.")
            dt = st["dt"]
            # Represent dwell as a jerk segment with j=0 for dt
            jerk_segs = [] if dt <= 0 else [{"name": "wait", "j": 0.0, "dt": dt}]
            plans.append({
                "type": "wait",
                "T": dt,
                "durations": (dt, 0.0, 0.0),
                "v_profile": (0.0, 0.0, 0.0),
                "jerk_segments": jerk_segs,
                "summary": f"WAIT: t={dt:.6g}s at x={x_curr:.6g} m (v=0).",
            })
            # x_curr unchanged; v_curr remains 0

    # Boundaries/time
    boundaries = [0.0]
    T_accum = 0.0
    for plan in plans:
        T_accum += plan["T"]
        boundaries.append(T_accum)

    # Human summary using processed + plans
    lines = ["MULTI-PROFILE PLAN:"]
    x_trace, v_trace = x0, v0
    for idx, (st, plan) in enumerate(zip(processed, plans), start=1):
        if st["type"] == "wait":
            lines.append(f"  Step {idx}: WAIT @ x={x_trace:.6g}  t={plan['T']:.6g}s")
        else:
            a_pref = st.get("a_pref")
            j_pref = st.get("j_pref")
            x_end = st["xf"]
            vf_end = st["vf"]
            cap = min(vmax, st["vcap"])
            mode = "S-curve" if plan.get("jerk_segments") is not None else "trapezoid"
            meta = []
            meta.append(f"cap={cap:.6g}")
            if a_pref is not None:
                meta.append(f"a_pref={a_pref:g}")
            if j_pref is not None:
                meta.append(f"j_pref={j_pref:g}")
            lines.append(
                f"  Step {idx}: MOVE x:{x_trace:.6g}->{x_end:.6g}  "
                f"v:{v_trace:.6g}->{vf_end:.6g}  ({', '.join(meta)})  "
                f"type={mode}  T={plan['T']:.6g}s"
            )
            x_trace, v_trace = x_end, vf_end
    lines.append(f"TOTAL time T={T_accum:.6g} s, steps={len(plans)}")

    return {
        "plans": plans,
        "segment_boundaries": boundaries,  # [0, t1, t2, ..., T]
        "T": T_accum,
        "summary": "\n".join(lines),
        "v_profile": (v0, v_trace, v_trace),
        "final_x": x_trace,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis
# ──────────────────────────────────────────────────────────────────────────────
def _synthesize_from_jerk(p0, v0, segments, samples=1200):
    """
    Integrate a list of jerk segments [{j, dt}] exactly, assuming a(0)=0.
    Outputs arrays t, x, v, a, j.
    """
    t0 = 0.0
    x = p0
    v = v0
    a = 0.0
    starts = []  # (t0, x0, v0, a0, j, dt)
    for seg in segments or []:
        dt = float(seg["dt"])
        if dt <= 0:
            continue
        j = float(seg["j"])
        starts.append((t0, x, v, a, j, dt))
        # advance to end of seg
        a_end = a + j * dt
        v_end = v + a * dt + 0.5 * j * dt * dt
        x_end = x + v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt ** 3
        t0, x, v, a = t0 + dt, x_end, v_end, a_end

    T = t0
    if T <= 0:
        t = np.array([0.0])
        return {"t": t, "x": np.full_like(t, p0), "v": np.full_like(t, v0), "a": np.zeros_like(t), "j": np.zeros_like(t)}

    n = max(int(samples), 3)
    t = np.linspace(0.0, T, n)
    x_arr = np.zeros_like(t)
    v_arr = np.zeros_like(t)
    a_arr = np.zeros_like(t)
    j_arr = np.zeros_like(t)

    bounds = np.cumsum([0.0] + [seg["dt"] for seg in segments or []])
    idx = 0
    for i, ti in enumerate(t):
        while idx + 1 < len(bounds) and ti > bounds[idx + 1] + 1e-15:
            idx += 1
        t0s, xs, vs, a0, j, dt_seg = starts[idx]
        dt = ti - t0s
        a_arr[i] = a0 + j * dt
        v_arr[i] = vs + a0 * dt + 0.5 * j * dt * dt
        x_arr[i] = xs + vs * dt + 0.5 * a0 * dt * dt + (1.0 / 6.0) * j * dt ** 3
        j_arr[i] = j

    return {"t": t, "x": x_arr, "v": v_arr, "a": a_arr, "j": j_arr}

def _synthesize_trapezoid_segment(p0, plan, samples=400):
    ta, tc, td = plan["durations"]
    T = ta + tc + td
    if T <= 0:
        t = np.array([0.0])
        return {"t": t, "x": np.full_like(t, p0), "v": np.zeros_like(t), "a": np.zeros_like(t), "j": np.zeros_like(t)}

    n = max(int(samples), 3)
    t = np.linspace(0.0, T, n)

    sgn = plan["sgn"]
    a_mag = abs(plan["a"][0])
    d_mag = abs(plan["a"][2])
    v0s = plan["v0s"]
    vmax_eff = plan["vmax_eff"]

    s = np.zeros_like(t)
    v = np.zeros_like(t)
    a_sig = np.zeros_like(t)

    t1 = ta
    t2 = ta + tc
    for i, ti in enumerate(t):
        if ti <= t1 + 1e-15:
            v[i] = v0s + a_mag * ti
            s[i] = v0s * ti + 0.5 * a_mag * ti * ti
            a_sig[i] = sgn * a_mag
        elif ti <= t2 + 1e-15:
            dt = ti - t1
            s[i] = plan["da"] + vmax_eff * dt
            v[i] = vmax_eff
            a_sig[i] = 0.0
        else:
            dt = ti - t2
            v_start = vmax_eff
            s_start = plan["da"] + vmax_eff * (t2 - t1)
            v[i] = max(v_start - d_mag * dt, 0.0)
            s[i] = s_start + v_start * dt - 0.5 * d_mag * dt * dt
            a_sig[i] = -sgn * d_mag

    x = p0 + sgn * s
    v_signed = sgn * v
    j = np.zeros_like(t)
    return {"t": t, "x": x, "v": v_signed, "a": a_sig, "j": j}

def synthesize_multiprofile(x0, v0, plans, samples=2000):
    """
    Concatenate per-step profiles (jerk, trapezoid, or wait).
    Samples distributed proportionally to step durations.
    """
    seg_T = [max(p["T"], 0.0) for p in plans]
    T_total = sum(seg_T)
    if T_total <= 0:
        t = np.array([0.0])
        return {"t": t, "x": np.full_like(t, x0), "v": np.full_like(t, v0), "a": np.zeros_like(t), "j": np.zeros_like(t)}

    # allocate samples per segment
    base = [max(3, int(round(samples * (Ti / T_total)))) for Ti in seg_T]
    diff = samples - sum(base)
    if diff != 0:
        base[-1] += diff

    t_all = []; x_all = []; v_all = []; a_all = []; j_all = []
    t_offset = 0.0
    x_state = x0
    v_state = v0

    for plan, n_i in zip(plans, base):
        if plan["type"] == "wait" or plan.get("jerk_segments") is not None:
            # jerk-based synthesis (includes wait with j=0)
            data = _synthesize_from_jerk(x_state, v_state, plan.get("jerk_segments", []), samples=n_i)
        else:
            data = _synthesize_trapezoid_segment(x_state, plan, samples=n_i)

        # stitch (skip first sample except for the first segment)
        slice_from = 1 if len(t_all) > 0 and len(data["t"]) > 0 else 0
        t_all.append(t_offset + data["t"][slice_from:])
        x_all.append(data["x"][slice_from:])
        v_all.append(data["v"][slice_from:])
        a_all.append(data["a"][slice_from:])
        j_all.append(data["j"][slice_from:])

        if len(data["t"]) > 0:
            t_offset += data["t"][-1]
            x_state = data["x"][-1]
            v_state = data["v"][-1]

    t = np.concatenate(t_all) if t_all else np.array([0.0])
    x = np.concatenate(x_all) if x_all else np.full_like(t, x0)
    v = np.concatenate(v_all) if x_all else np.full_like(t, v0)
    a = np.concatenate(a_all) if x_all else np.zeros_like(t)
    j = np.concatenate(j_all) if x_all else np.zeros_like(t)
    return {"t": t, "x": x, "v": v, "a": a, "j": j}


# ──────────────────────────────────────────────────────────────────────────────
# Tkinter App
# ──────────────────────────────────────────────────────────────────────────────
class MotionApp(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Motion Profile Designer — Move/Wait Multiprofile (S-Curve / Trapezoid)")
        self.master.geometry("1420x880")
        self.master.minsize(1150, 700)

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_form(self)
        self._build_plot(self)
        self._set_defaults()

    def _build_form(self, parent):
        frm = ttk.Frame(parent, padding=(12, 10))
        frm.grid(row=0, column=0, sticky="nsw")

        lbl = ttk.Label(frm, text="Inputs", font=("Segoe UI", 12, "bold"))
        lbl.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.vars = {
            "x0": tk.StringVar(value="0.0"),
            "xf": tk.StringVar(value="0.0"),
            "v0": tk.StringVar(value="0.0"),
            "vf": tk.StringVar(value="0.0"),
            "vmax": tk.StringVar(value="100.0"),
            "amax": tk.StringVar(value="100.0"),
            "dmax": tk.StringVar(value="100.0"),
            "jmax": tk.StringVar(value="1000.0"),
            "samples": tk.StringVar(value="2000"),
        }

        def add_row(r, label, key, unit):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            ent = ttk.Entry(frm, textvariable=self.vars[key], width=16, justify="right")
            ent.grid(row=r, column=1, sticky="w", pady=4)
            ttk.Label(frm, text=unit, foreground="#666").grid(row=r, column=2, sticky="w", padx=(6, 0))

        add_row(1,  "Start position x₀", "x0", "m")
        add_row(2,  "Final position x_f (single)", "xf", "m")
        ttk.Separator(frm, orient="horizontal").grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 6))
        add_row(4,  "Initial velocity v₀", "v0", "m/s")
        add_row(5,  "Final velocity v_f (single)", "vf", "m/s")
        add_row(6,  "Max velocity v_max", "vmax", "m/s")
        add_row(7,  "Max acceleration a_max", "amax", "m/s²")
        add_row(8,  "Max deceleration d_max", "dmax", "m/s²")
        add_row(9,  "Max jerk j_max", "jmax", "m/s³")
        ttk.Separator(frm, orient="horizontal").grid(row=10, column=0, columnspan=3, sticky="ew", pady=(6, 6))

        # Steps table
        mp = ttk.LabelFrame(
            frm,
            text=(
                "Steps (≤10): choose a Step type per row — Move or Wait.\n"
                "Move: Target x [m], Target v (speed) [m/s], Target a (opt) [m/s²], Jerk j (opt) [m/s³].\n"
                "  • Target v is the speed AT the waypoint (magnitude); its sign is inferred from Δx and it also caps the\n"
                "    segment peak speed (v_peak ≤ min(v_max, |Target v|)).\n"
                "  • Target a is signed and only tightens the active accel limit (+ when speeding up, − when slowing down).\n"
                "  • Jerk j (optional) sets a per-step cap J_step = min(|j|, j_max). Leave blank to use global j_max; j ≤ 0 disables\n"
                "    the jerk limit for that step (trapezoid fallback).\n"
                "Wait: dwell t [s] at the current x with v=0. The previous Move is forced to end at v=0. Direction reversals and\n"
                "Waits automatically insert a stop at the preceding waypoint."
            )
        )

        mp.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(2, 8))
        for c in range(10):
            mp.columnconfigure(c, weight=0)

        ttk.Label(mp, text="#").grid(row=0, column=0, padx=2)
        ttk.Label(mp, text="Use").grid(row=0, column=1, padx=2)
        ttk.Label(mp, text="Type").grid(row=0, column=2, padx=2)
        ttk.Label(mp, text="Target x [m]").grid(row=0, column=3, padx=2)
        ttk.Label(mp, text="Target v (speed) [m/s]").grid(row=0, column=4, padx=2)
        ttk.Label(mp, text="Target a (opt) [m/s²]").grid(row=0, column=5, padx=2)
        ttk.Label(mp, text="Jerk j (opt) [m/s³]").grid(row=0, column=6, padx=2)
        ttk.Label(mp, text="Wait t [s]").grid(row=0, column=7, padx=2)

        self.mp_rows = []
        for i in range(10):
            r = i + 1
            enabled = tk.BooleanVar(value=False)
            type_var = tk.StringVar(value="Move")
            x_var = tk.StringVar(value="")
            v_var = tk.StringVar(value="")
            a_var = tk.StringVar(value="")
            j_var = tk.StringVar(value="")  # per-step jerk
            t_var = tk.StringVar(value="")  # dwell time

            ttk.Label(mp, text=f"{i+1}").grid(row=r, column=0, padx=2)
            ttk.Checkbutton(mp, variable=enabled).grid(row=r, column=1, padx=2)

            cmb = ttk.Combobox(mp, textvariable=type_var, values=["Move", "Wait"], width=7, state="readonly")
            cmb.grid(row=r, column=2, padx=2, pady=1)

            e_x = ttk.Entry(mp, textvariable=x_var, width=12, justify="right")
            e_v = ttk.Entry(mp, textvariable=v_var, width=12, justify="right")
            e_a = ttk.Entry(mp, textvariable=a_var, width=12, justify="right")
            e_j = ttk.Entry(mp, textvariable=j_var, width=12, justify="right")
            e_t = ttk.Entry(mp, textvariable=t_var, width=10, justify="right")
            e_x.grid(row=r, column=3, padx=2, pady=1)
            e_v.grid(row=r, column=4, padx=2, pady=1)
            e_a.grid(row=r, column=5, padx=2, pady=1)
            e_j.grid(row=r, column=6, padx=2, pady=1)
            e_t.grid(row=r, column=7, padx=2, pady=1)

            def make_on_type_change(entries_move=(e_x, e_v, e_a, e_j), entry_wait=e_t, var=type_var):
                def _on_change(*_):
                    if var.get().lower() == "wait":
                        for w in entries_move:
                            w.configure(state="disabled")
                        entry_wait.configure(state="normal")
                    else:
                        for w in entries_move:
                            w.configure(state="normal")
                        entry_wait.configure(state="disabled")
                return _on_change

            handler = make_on_type_change()
            type_var.trace_add("write", lambda *_h, fn=handler: fn())
            handler()  # initial state

            self.mp_rows.append({
                "enabled": enabled, "type": type_var,
                "xf": x_var, "vf": v_var, "af": a_var, "jf": j_var, "wt": t_var
            })

        ttk.Separator(frm, orient="horizontal").grid(row=12, column=0, columnspan=3, sticky="ew", pady=(6, 6))
        add_row(13, "Samples", "samples", "points")

        self.use_latex_var = tk.BooleanVar(value=HAS_LATEX)
        ttk.Checkbutton(frm, text="Use LaTeX (requires TeX)", variable=self.use_latex_var)\
            .grid(row=14, column=0, columnspan=3, sticky="w", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=15, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(btns, text="Compute & Plot", command=self.on_compute).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Export CSV…",   command=self.on_export_csv).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(btns, text="Save Plot…",    command=self.on_save_plot).grid(row=0, column=2, padx=(0, 8))

        ttk.Separator(frm, orient="horizontal").grid(row=16, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        ttk.Label(frm, text="Summary", font=("Segoe UI", 11, "bold")).grid(row=17, column=0, columnspan=3, sticky="w")
        self.txt_summary = tk.Text(frm, width=64, height=18, wrap="word", relief="solid", borderwidth=1)
        self.txt_summary.grid(row=18, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        for c in range(3):
            frm.columnconfigure(c, weight=0)

    def _build_plot(self, parent):
        right = ttk.Frame(parent, padding=(8, 8, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        toggles = ttk.Frame(right)
        toggles.grid(row=0, column=0, sticky="w")
        self.show_pos  = tk.BooleanVar(value=True)
        self.show_vel  = tk.BooleanVar(value=True)
        self.show_acc  = tk.BooleanVar(value=True)
        self.show_jerk = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggles, text="Position",     variable=self.show_pos,  command=self.redraw).grid(row=0, column=0, padx=(0, 8))
        ttk.Checkbutton(toggles, text="Velocity",     variable=self.show_vel,  command=self.redraw).grid(row=0, column=1, padx=(0, 8))
        ttk.Checkbutton(toggles, text="Acceleration", variable=self.show_acc,  command=self.redraw).grid(row=0, column=2, padx=(0, 8))
        ttk.Checkbutton(toggles, text="Jerk",         variable=self.show_jerk, command=self.redraw).grid(row=0, column=3, padx=(0, 8))

        self.fig = Figure(figsize=(7.8, 5.8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(r"Position--Time Profile $x(t)$")
        self.ax.set_xlabel(r"Time $t$ [s]")
        self.ax.set_ylabel(r"Position $x$ [m]")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        self.fig.tight_layout(pad=1.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=2, column=0, sticky="ew")

        self._last_data = None

    def _set_defaults(self):
        self.on_compute()

    def _parse_floats(self):
        try:
            x0 = float(self.vars["x0"].get())
            xf = float(self.vars["xf"].get())
            v0 = float(self.vars["v0"].get())
            vf = float(self.vars["vf"].get())
            vmax = float(self.vars["vmax"].get())
            amax = float(self.vars["amax"].get())
            dmax = float(self.vars["dmax"].get())
            jmax = float(self.vars["jmax"].get())
            samples = int(float(self.vars["samples"].get()))
            if samples < 50:
                samples = 50

            steps = []
            for row in self.mp_rows:
                if not row["enabled"].get():
                    continue
                rtype = row["type"].get().strip().lower()
                if rtype == "move":
                    x_str = row["xf"].get().strip()
                    v_str = row["vf"].get().strip()
                    a_str = row["af"].get().strip()
                    j_str = row["jf"].get().strip()
                    if x_str == "" or v_str == "":
                        raise ValueError("Enabled Move rows must have both Target x and Target v filled.")
                    a_pref = None if a_str == "" else float(a_str)
                    j_pref = None if j_str == "" else float(j_str)
                    steps.append({"type":"move", "xf": float(x_str), "v_speed": float(v_str),
                                  "a_pref": a_pref, "j_pref": j_pref})
                elif rtype == "wait":
                    t_str = row["wt"].get().strip()
                    if t_str == "":
                        raise ValueError("Enabled Wait rows must have Wait t filled.")
                    steps.append({"type":"wait", "dt": float(t_str)})
                else:
                    raise ValueError("Unknown step type.")
            return x0, xf, v0, vf, vmax, amax, dmax, jmax, samples, steps
        except ValueError:
            raise

    def on_compute(self):
        try:
            x0, xf, v0, vf, vmax, amax, dmax, jmax, samples, steps = self._parse_floats()

            if len(steps) > 0:
                # Multi-profile path (Move/Wait)
                mp = plan_multiprofile(x0, v0, steps, vmax, amax, dmax, jmax)
                data = synthesize_multiprofile(x0, v0, mp["plans"], samples=samples)
                self._last_data = {
                    "x0": x0,
                    "xf": mp.get("final_x", x0),
                    "durations": (mp["T"], 0.0, 0.0),
                    "T": mp["T"],
                    "v_profile": mp["v_profile"],
                    "segment_boundaries": mp["segment_boundaries"],
                    "summary": mp["summary"],
                    **data,
                }
            else:
                # Single segment (use xf,vf fields)
                prof = plan_profile(x0, xf, v0, vf, vmax, amax, dmax, jmax, v_cap=None)
                data = (_synthesize_trapezoid_segment(x0, prof, samples=samples)
                        if prof.get("jerk_segments") is None
                        else _synthesize_from_jerk(x0, prof["v_profile"][0], prof["jerk_segments"], samples=samples))
                self._last_data = {"x0": x0, "xf": xf, **prof, **data}

            self._update_summary(self._last_data["summary"])
            self.redraw()
        except Exception as e:
            messagebox.showerror("Error", f"{type(e).__name__}: {e}")

    def _update_summary(self, text):
        self.txt_summary.config(state="normal")
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("end", text + "\n")
        self.txt_summary.config(state="disabled")

    def redraw(self):
        self.ax.clear()
        # remove any old twins
        for extra_ax in list(self.fig.axes):
            if extra_ax is not self.ax:
                extra_ax.remove()

        use_tex = bool(self.use_latex_var.get())
        try:
            matplotlib.rcParams["text.usetex"] = use_tex and HAS_LATEX
        except Exception:
            matplotlib.rcParams["text.usetex"] = False

        if self._last_data is None:
            self.ax.set_title(r"Position--Time Profile $x(t)$")
            self.ax.set_xlabel(r"Time $t$ [s]")
            self.ax.set_ylabel(r"Position $x$ [m]")
            self.canvas.draw_idle()
            return

        t = self._last_data["t"]
        x = self._last_data["x"]
        v = self._last_data["v"]
        a = self._last_data["a"]
        j = self._last_data.get("j", np.zeros_like(t))

        bounds = self._last_data.get("segment_boundaries")
        T = (bounds[-1] if bounds else sum(self._last_data.get("durations", (0, 0, 0))))

        lines, labels = [], []

        if self.show_pos.get():
            ln, = self.ax.plot(t, x, linewidth=2.0, label=r"$x(t)$", color=COLORS["pos"])
            self.ax.set_ylabel(r"Position $x$ [m]", color=COLORS["pos"])
            self.ax.tick_params(axis="y", colors=COLORS["pos"])
            lines.append(ln); labels.append(r"$x(t)$")

        if self.show_vel.get():
            ax2 = self.ax.twinx()
            ln2, = ax2.plot(t, v, linestyle="--", linewidth=1.5, label=r"$v(t)$", color=COLORS["vel"])
            ax2.set_ylabel(r"Velocity $v$ [m/s]", color=COLORS["vel"])
            ax2.tick_params(axis="y", colors=COLORS["vel"])
            ax2.grid(False)
            lines.append(ln2); labels.append(r"$v(t)$")

        if self.show_acc.get():
            ax3 = self.ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.12))
            ln3, = ax3.plot(t, a, linestyle="-.", linewidth=1.2, label=r"$a(t)$", color=COLORS["acc"])
            ax3.set_ylabel(r"Acceleration $a$ [m/s^2]", color=COLORS["acc"])
            ax3.tick_params(axis="y", colors=COLORS["acc"])
            ax3.grid(False)
            lines.append(ln3); labels.append(r"$a(t)$")

        if self.show_jerk.get():
            ax4 = self.ax.twinx()
            ax4.spines["right"].set_position(("axes", 1.24))
            ln4, = ax4.plot(t, j, linestyle=":", linewidth=1.2, label=r"$\dot a(t)$", color=COLORS["jerk"])
            ax4.set_ylabel(r"Jerk $\dot a$ [m/s^3]", color=COLORS["jerk"])
            ax4.tick_params(axis="y", colors=COLORS["jerk"])
            ax4.grid(False)
            lines.append(ln4); labels.append(r"$\dot a(t)$")

        self.ax.set_title(r"Position--Time Profile $x(t)$")
        self.ax.set_xlabel(r"Time $t$ [s]")
        self.ax.set_ylabel(r"Position $x$ [m]")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        self.ax.grid(True, which="both", axis="both", alpha=0.25)

        # draw step boundaries for multiprofile
        if bounds:
            for tx in bounds[1:-1]:
                if 0 < tx < T:
                    self.ax.axvline(tx, color="k", alpha=0.15, linestyle=":")

        if lines:
            self.ax.legend(lines, labels, loc="upper left")

        self.fig.tight_layout(pad=1.1)
        try:
            self.canvas.draw_idle()
        except Exception:
            matplotlib.rcParams["text.usetex"] = False
            self.canvas.draw_idle()

    def on_export_csv(self):
        if self._last_data is None:
            messagebox.showinfo("Nothing to export", "Please compute a profile first.")
            return
        fpath = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="motion_profile.csv",
        )
        if not fpath:
            return
        try:
            with open(fpath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t [s]", "x [m]", "v [m/s]", "a [m/s^2]", "j [m/s^3]"])
                j = self._last_data.get("j", np.zeros_like(self._last_data["t"]))
                for ti, xi, vi, ai, ji in zip(self._last_data["t"], self._last_data["x"],
                                              self._last_data["v"], self._last_data["a"], j):
                    writer.writerow([f"{ti:.9f}", f"{xi:.9f}", f"{vi:.9f}", f"{ai:.9f}", f"{ji:.9f}"])
            messagebox.showinfo("Exported", f"Saved CSV to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Failed", f"Could not write CSV:\n{e}")

    def on_save_plot(self):
        if self._last_data is None:
            messagebox.showinfo("Nothing to save", "Please compute a profile first.")
            return
        fpath = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="motion_profile.png",
        )
        if not fpath:
            return
        try:
            self.fig.savefig(fpath, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Saved plot to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Failed", f"Could not save plot:\n{e}")


def main():
    root = tk.Tk()
    app = MotionApp(master=root)
    app.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
