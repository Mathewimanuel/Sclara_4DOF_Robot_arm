"""
IKSolver.py
===========
Inverse Kinematics solver for 4-DOF desktop robot arm.
All conventions match ARM_IK_CONVENTIONS_v4.

Architecture: runs on LAPTOP only. ESP32 is a pure motor controller.

World frame:
  Origin : table surface, directly below shoulder pivot (base rotation axis)
  +X     : right
  +Y     : forward
  +Z     : up

Joint IK zeros:
  BASE     : pointing +Y (forward)
  SHOULDER : 90deg from horizontal (exactly vertical)
  ELBOW    : 20deg interior angle (fully tucked)
  WRIST    : horizontal / forward (auto-derived, never free variable)

Author  : Robot Arm Project
Version : 1.0  (matches conventions v4)
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  —  all match ARM_IK_CONVENTIONS_v4
# ═══════════════════════════════════════════════════════════════════════════════

# ── Physical dimensions (mm) ──────────────────────────────────────────────────
L1                   = 160.0   # shoulder pivot → elbow pivot
L2                       = 138.0   # elbow pivot → wrist pivot straight-line (measured)
L2_PERP                  = 13.0    # wrist pivot perpendicular offset above forearm axis (measured)
L2_AXIS              = math.sqrt(L2**2 - L2_PERP**2)   # derived from measured L2 and L2_PERP
TOOL_MM                  = 118.6   # wrist pivot → tool tip horizontal (measured)
TOOL_Z_OFFSET_MM      = -7.3    # wrist pivot → tool tip vertical offset (tool tip sits below pivot)
FOREARM_EXTENSION_MM = 25.0    # wrist pivot to mounting plate
TOOL_FROM_PLATE_MM   = 50.0    # mounting plate to tool tip

SHOULDER_Z_OFFSET_MM     = 146.0   # table surface → shoulder pivot height (measured)
BASE_MOUNT_HEIGHT_MM  = 95.0   # table surface → top of mount cylinder
SHOULDER_DH_OFFSET_MM    = 38.0   # base rotation axis → shoulder pivot horizontal offset (measured)

# ── DH offset angle (constant, computed once) ─────────────────────────────────
# Wrist pivot is 20mm above elbow in forearm frame → effective L2 rotated by BETA
BETA = math.atan2(L2_PERP, L2_AXIS)   # ≈ 5.41° (derived from L2=138, L2_PERP=13)

# ── Wrist structure ───────────────────────────────────────────────────────────
WRIST_HALF_HEIGHT_MM = 17.5    # half of 35mm body → bottom hits table at pivot Z < 17.5mm
WRIST_MOUNT_OFFSET   = 0.0     # tune empirically after assembly

# ── Gear ratios ───────────────────────────────────────────────────────────────
BASE_GEAR_RATIO      = 5.0
SHOULDER_GEAR_RATIO  = 20.0
ELBOW_GEAR_RATIO     = 5.0
WRIST_GEAR_RATIO     = 1.6     # 32/20 teeth EXACT — not 1.61

STEPS_PER_REV        = 200     # full step, no microstepping

# Steps per degree for each joint
BASE_STEPS_PER_DEG     = (STEPS_PER_REV * BASE_GEAR_RATIO)     / 360.0   # 13.889
SHOULDER_STEPS_PER_DEG = (STEPS_PER_REV * SHOULDER_GEAR_RATIO) / 360.0   # 11.111  (wait — 4000/360)
ELBOW_STEPS_PER_DEG    = (STEPS_PER_REV * ELBOW_GEAR_RATIO)    / 360.0   # 2.778
WRIST_STEPS_PER_DEG    = (STEPS_PER_REV * WRIST_GEAR_RATIO)    / 360.0   # 0.889

# ── Soft limits (degrees from IK zero) ───────────────────────────────────────
BASE_MIN_DEG      = -110.0
BASE_MAX_DEG      =  110.0
SHOULDER_MIN_DEG  =    0.0
SHOULDER_MAX_DEG  =   98.0
ELBOW_MIN_DEG     = -115.0   # negative = opening arm (normal operation)
ELBOW_MAX_DEG     =    0.0   # cannot tuck past IK zero
WRIST_MIN_DEG     = -149.0   # physical travel limit (180° - 31° park offset)
WRIST_MAX_DEG     =   26.0   # physical travel limit (31° park - 5° switch release)

# ── Collision zones ───────────────────────────────────────────────────────────
MOUNT_RADIUS_MM         = 40.0
SHOULDER_SWEEP_RADIUS_MM = 78.0   # DH_OFFSET(38) + pulley_radius(40) (updated)
COLLISION_CLEARANCE_MM  = 10.0

# ── Solver ────────────────────────────────────────────────────────────────────
SINGULARITY_THRESHOLD_MM = 5.0
LIMIT_TOLERANCE_DEG      = 0.5    # floating point tolerance on limit checks


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IKResult:
    # Joint deltas from IK zero (degrees)
    base_delta:     float = 0.0
    shoulder_delta: float = 0.0
    elbow_delta:    float = 0.0
    wrist_delta:    float = 0.0

    # Absolute step targets from IK zero
    base_steps:     int   = 0
    shoulder_steps: int   = 0
    elbow_steps:    int   = 0
    wrist_steps:    int   = 0

    # Status
    reachable: bool = True
    warnings: list  = field(default_factory=list)

    def serial_string(self) -> str:
        """Format as serial command string for ESP32."""
        return f"B:{self.base_steps} S:{self.shoulder_steps} E:{self.elbow_steps} W:{self.wrist_steps}"

    def __str__(self) -> str:
        lines = [
            f"  Base:     {self.base_delta:+7.2f}°  →  {self.base_steps:+6d} steps",
            f"  Shoulder: {self.shoulder_delta:+7.2f}°  →  {self.shoulder_steps:+6d} steps",
            f"  Elbow:    {self.elbow_delta:+7.2f}°  →  {self.elbow_steps:+6d} steps",
            f"  Wrist:    {self.wrist_delta:+7.2f}°  →  {self.wrist_steps:+6d} steps",
            f"  Reachable: {self.reachable}",
            f"  Serial:    {self.serial_string()}",
        ]
        if self.warnings:
            lines.append(f"  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


@dataclass
class FKResult:
    x: float   # tool tip X (mm)
    y: float   # tool tip Y (mm)
    z: float   # tool tip Z (mm)
    shoulder_world_deg: float   # shoulder angle from horizontal
    elbow_interior_deg: float   # elbow interior angle
    forearm_angle_deg:  float   # forearm absolute angle from horizontal

    def __str__(self) -> str:
        return (
            f"  Tool tip:  X={self.x:+7.2f}  Y={self.y:+7.2f}  Z={self.z:+7.2f} mm\n"
            f"  Shoulder world: {self.shoulder_world_deg:.2f}° from horizontal\n"
            f"  Elbow interior: {self.elbow_interior_deg:.2f}°\n"
            f"  Forearm angle:  {self.forearm_angle_deg:.2f}° from horizontal"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _deg(r: float) -> float:
    return math.degrees(r)

def _rad(d: float) -> float:
    return math.radians(d)


# ═══════════════════════════════════════════════════════════════════════════════
# COLLISION CHECKING
# ═══════════════════════════════════════════════════════════════════════════════

def _check_point(x: float, y: float, z: float, label: str):
    """
    Check a single world-frame point against all exclusion zones.
    Returns (hit: bool, zone: str, msg: str)
    """
    r = math.sqrt(x**2 + y**2)
    C = COLLISION_CLEARANCE_MM

    # Zone 3: table floor
    # Wrist structure is 35mm tall, pivot centered → bottom is 17.5mm below pivot
    # Minimum Z for wrist pivot (and tool tip at same Z) = 17.5mm
    if z < WRIST_HALF_HEIGHT_MM:
        return True, "TABLE", (
            f"{label} too low — Z={z:.1f}mm "
            f"(min {WRIST_HALF_HEIGHT_MM}mm, bottom of wrist structure hits table)"
        )

    # Zone 1: mount cylinder
    if z < BASE_MOUNT_HEIGHT_MM and r < MOUNT_RADIUS_MM + C:
        return True, "ZONE1", (
            f"{label} inside mount cylinder — "
            f"r={r:.0f}mm (min {MOUNT_RADIUS_MM + C:.0f}mm at Z={z:.0f}mm)"
        )

    # Zone 2: shoulder sweep zone
    if BASE_MOUNT_HEIGHT_MM <= z < SHOULDER_Z_OFFSET_MM and r < SHOULDER_SWEEP_RADIUS_MM + C:
        return True, "ZONE2", (
            f"{label} inside shoulder sweep — "
            f"r={r:.0f}mm (min {SHOULDER_SWEEP_RADIUS_MM + C:.0f}mm at Z={z:.0f}mm)"
        )

    return False, "", ""


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD KINEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArmGeometry:
    """World-frame XYZ positions of all arm joints for visualisation."""
    base:     tuple   # (x, y, z) base rotation axis at table
    shoulder: tuple   # (x, y, z) shoulder pivot
    elbow:    tuple   # (x, y, z) elbow pivot
    wrist:    tuple   # (x, y, z) wrist pivot
    tip:      tuple   # (x, y, z) tool tip
    target:   tuple   # (x, y, z) target position


def arm_geometry(base_delta: float, shoulder_delta: float,
                 elbow_delta: float, target_x: float,
                 target_y: float, target_z: float) -> ArmGeometry:
    """
    Compute world-frame XYZ for all arm joints — for 3D visualisation.
    """
    base_rad  = _rad(base_delta)
    alpha_rad = _rad(90.0 - shoulder_delta)
    theta_deg = 20.0 - elbow_delta
    phi_rad   = alpha_rad - math.pi + _rad(theta_deg)

    arm_x = -math.sin(base_rad)
    arm_y =  math.cos(base_rad)

    sh_x = -SHOULDER_DH_OFFSET_MM * math.sin(base_rad)
    sh_y =  SHOULDER_DH_OFFSET_MM * math.cos(base_rad)
    sh_z =  SHOULDER_Z_OFFSET_MM

    e_r  = L1 * math.cos(alpha_rad)
    e_z  = L1 * math.sin(alpha_rad)
    el_x = sh_x + e_r * arm_x
    el_y = sh_y + e_r * arm_y
    el_z = sh_z + e_z

    w_r  = L2 * math.cos(phi_rad + BETA)
    w_z  = L2 * math.sin(phi_rad + BETA)
    wr_x = el_x + w_r * arm_x
    wr_y = el_y + w_r * arm_y
    wr_z = el_z + w_z

    tp_x = wr_x + TOOL_MM * arm_x
    tp_y = wr_y + TOOL_MM * arm_y
    tp_z = wr_z + TOOL_Z_OFFSET_MM

    return ArmGeometry(
        base     = (0.0,  0.0,  0.0),
        shoulder = (sh_x, sh_y, sh_z),
        elbow    = (el_x, el_y, el_z),
        wrist    = (wr_x, wr_y, wr_z),
        tip      = (tp_x, tp_y, tp_z),
        target   = (target_x, target_y, target_z),
    )


def forward(base_delta: float, shoulder_delta: float, elbow_delta: float) -> FKResult:
    """
    Forward kinematics — given joint deltas from IK zero, compute tool tip world position.

    Args:
        base_delta:     degrees from IK zero, positive = left (CCW from above)
        shoulder_delta: degrees from IK zero, positive = forward/down
        elbow_delta:    degrees from IK zero, positive = tuck, negative = open (normal)

    Returns:
        FKResult with tool tip XYZ in world frame (table origin)
    """
    base_rad    = _rad(base_delta)
    alpha_rad   = _rad(90.0 - shoulder_delta)
    theta_deg   = 20.0 - elbow_delta
    phi_rad     = alpha_rad - math.pi + _rad(theta_deg)

    arm_x = -math.sin(base_rad)
    arm_y =  math.cos(base_rad)

    sh_x = -SHOULDER_DH_OFFSET_MM * math.sin(base_rad)
    sh_y =  SHOULDER_DH_OFFSET_MM * math.cos(base_rad)
    sh_z =  SHOULDER_Z_OFFSET_MM

    elbow_r = L1 * math.cos(alpha_rad)
    elbow_z = L1 * math.sin(alpha_rad)

    wrist_r = L2 * math.cos(phi_rad + BETA)
    wrist_z = L2 * math.sin(phi_rad + BETA)

    total_r = elbow_r + wrist_r + TOOL_MM

    return FKResult(
        x = sh_x + total_r * arm_x,
        y = sh_y + total_r * arm_y,
        z = sh_z + elbow_z + wrist_z + TOOL_Z_OFFSET_MM,
        shoulder_world_deg = _deg(alpha_rad),
        elbow_interior_deg = theta_deg,
        forearm_angle_deg  = _deg(phi_rad),
    )






# ═══════════════════════════════════════════════════════════════════════════════
# INVERSE KINEMATICS
# ═══════════════════════════════════════════════════════════════════════════════

def solve(x: float, y: float, z: float) -> IKResult:
    """
    Inverse kinematics solver.

    Input:  tool tip target in world frame (table origin, mm)
    Output: IKResult with joint deltas and absolute step targets

    Pipeline:
        Stage 1  — Collision check on tool tip
        Stage 2  — Base azimuth
        Stage 3  — Tool compensation (step back 75mm to wrist pivot target)
        Stage 4  — Frame conversion to shoulder-pivot frame
        Stage 5  — Singularity check
        Stage 6  — Reachability check
        Stage 7  — Cosine rule (2-link planar IK)
        Stage 8  — Joint angles
        Stage 9  — Wrist compensation
        Stage 10 — Soft limit checks
        Stage 11 — Joint position collision checks (elbow, wrist)
        Stage 12 — Convert to step counts
    """
    warnings = []
    reachable = True

    # ── STAGE 1: Collision check on tool tip ──────────────────────────────────
    hit, zone, msg = _check_point(x, y, z, "TIP")
    if hit:
        warnings.append(f"COLLISION [{zone}]: {msg}")
        reachable = False

    # ── STAGE 2: Base azimuth ─────────────────────────────────────────────────
    # atan2(-x, y) because positive base = left = CCW from above
    base_rad = math.atan2(-x, y)
    base_delta = _deg(base_rad)

    # ── STAGE 3: Tool compensation ────────────────────────────────────────────
    # Step back along base azimuth to get wrist pivot target
    # Also remove vertical tool offset to get wrist pivot Z
    wx = x + TOOL_MM * math.sin(base_rad)
    wy = y - TOOL_MM * math.cos(base_rad)
    wz = z - TOOL_Z_OFFSET_MM

    # ── STAGE 4: Frame conversion to shoulder-pivot frame ────────────────────
    # Shoulder pivot world position at this base angle
    sh_x = -SHOULDER_DH_OFFSET_MM * math.sin(base_rad)
    sh_y =  SHOULDER_DH_OFFSET_MM * math.cos(base_rad)

    # Wrist target relative to shoulder pivot
    dx = wx - sh_x
    dy = wy - sh_y
    r  = math.sqrt(dx**2 + dy**2)   # horizontal distance in arm plane
    dz = wz - SHOULDER_Z_OFFSET_MM  # vertical distance from shoulder pivot

    # ── STAGE 5: Singularity check ────────────────────────────────────────────
    D = math.sqrt(r**2 + dz**2)
    if D < SINGULARITY_THRESHOLD_MM:
        warnings.append(
            f"SINGULARITY: Target on shoulder axis (D={D:.1f}mm) — holding base angle"
        )
        reachable = False

    # ── STAGE 6: Reachability check ───────────────────────────────────────────
    D_use = D
    if D > L1 + L2:
        warnings.append(
            f"REACH: Too far — D={D:.1f}mm, max={L1+L2:.0f}mm — clamped"
        )
        D_use = L1 + L2 - 0.5
        reachable = False

    min_reach = abs(L1 - L2)
    if D < min_reach:
        warnings.append(
            f"REACH: Too close — D={D:.1f}mm, min={min_reach:.0f}mm — clamped"
        )
        D_use = min_reach + 0.5
        reachable = False

    # ── STAGE 7: Cosine rule ──────────────────────────────────────────────────
    # Phi = angle at elbow pivot in shoulder-elbow-wrist triangle
    cos_phi = _clamp((L1**2 + L2**2 - D_use**2) / (2 * L1 * L2), -1.0, 1.0)
    Phi = math.acos(cos_phi)

    # psi = angle at shoulder pivot in shoulder-elbow-wrist triangle
    cos_psi = _clamp((L1**2 + D_use**2 - L2**2) / (2 * L1 * D_use), -1.0, 1.0)
    psi = math.acos(cos_psi)

    # ── STAGE 8: Joint angles ─────────────────────────────────────────────────
    gamma     = math.atan2(dz, r)     # direction from shoulder to wrist target
    alpha_rad = gamma + psi           # shoulder angle from horizontal (elbow-up solution)
    alpha_deg = _deg(alpha_rad)

    # Elbow interior angle — subtract BETA to account for 20mm DH offset
    theta_deg = _deg(Phi) - _deg(BETA)

    # Convert to deltas from IK zero
    shoulder_delta = 90.0 - alpha_deg   # IK zero = 90° from horizontal
    elbow_delta    = 20.0 - theta_deg   # IK zero = 20° interior angle

    # ── STAGE 9: Wrist compensation ───────────────────────────────────────────
    # 70° geometric offset at IK zero cancels (step counter zeroed there)
    # Only joint movement deltas matter
    wrist_delta = shoulder_delta + elbow_delta + WRIST_MOUNT_OFFSET

    # Physical wrist travel check.
    # Limit switch is at mechanical zero; physical hard stop is 180° away.
    # IK zero (park position) is WRIST_PARK_DELTA_DEG (31°) from the switch.
    # From IK zero:
    #   Positive (away from switch):  180 - 31 = +149°
    #   Negative (toward switch):     -(31 - WRIST_SWITCH_RELEASE_DEG) ≈ -26°
    WRIST_PHYS_MIN = -149.0   # away from switch: 180° travel - 31° park offset
    WRIST_PHYS_MAX =   26.0   # toward switch:    31° park - 5° switch release backoff
    if wrist_delta > WRIST_PHYS_MAX or wrist_delta < WRIST_PHYS_MIN:
        warnings.append(
            f"WRIST PHYSICAL: Required compensation {wrist_delta:+.1f}° outside "
            f"physical travel [{WRIST_PHYS_MIN:.0f}°, +{WRIST_PHYS_MAX:.0f}°] — "
            f"gripper will not remain level"
        )
        reachable = False

    # ── STAGE 10: Soft limit checks ───────────────────────────────────────────
    def limit_check(name, val, lo, hi):
        nonlocal reachable
        if val < lo - LIMIT_TOLERANCE_DEG or val > hi + LIMIT_TOLERANCE_DEG:
            warnings.append(
                f"LIMIT [{name}]: {val:.1f}° outside [{lo}, {hi}]"
            )
            reachable = False
        return _clamp(val, lo, hi)

    base_delta_c     = limit_check("BASE",     base_delta,     BASE_MIN_DEG,     BASE_MAX_DEG)
    shoulder_delta_c = limit_check("SHOULDER", shoulder_delta, SHOULDER_MIN_DEG, SHOULDER_MAX_DEG)
    elbow_delta_c    = limit_check("ELBOW",    elbow_delta,    ELBOW_MIN_DEG,    ELBOW_MAX_DEG)
    wrist_delta_c    = limit_check("WRIST",    wrist_delta,    WRIST_MIN_DEG,    WRIST_MAX_DEG)

    # ── STAGE 11: Joint position collision checks ─────────────────────────────
    # Compute arm geometry for collision checking
    arm_x_dir = -math.sin(base_rad)
    arm_y_dir =  math.cos(base_rad)

    elbow_r_loc = L1 * math.cos(alpha_rad)
    elbow_z_loc = L1 * math.sin(alpha_rad)

    phi_rad     = alpha_rad - math.pi + _rad(theta_deg)
    wrist_r_loc = elbow_r_loc + L2 * math.cos(phi_rad + BETA)
    wrist_z_loc = elbow_z_loc + L2 * math.sin(phi_rad + BETA)

    # Elbow world position
    el_x = sh_x + elbow_r_loc * arm_x_dir
    el_y = sh_y + elbow_r_loc * arm_y_dir
    el_z = SHOULDER_Z_OFFSET_MM + elbow_z_loc
    hit, zone, msg = _check_point(el_x, el_y, el_z, "ELBOW")
    if hit:
        warnings.append(f"COLLISION [{zone}]: {msg}")
        reachable = False

    # Wrist world position
    wr_x = sh_x + wrist_r_loc * arm_x_dir
    wr_y = sh_y + wrist_r_loc * arm_y_dir
    wr_z = SHOULDER_Z_OFFSET_MM + wrist_z_loc
    hit, zone, msg = _check_point(wr_x, wr_y, wr_z, "WRIST")
    if hit:
        warnings.append(f"COLLISION [{zone}]: {msg}")
        reachable = False

    # ── STAGE 12: Convert to step counts ─────────────────────────────────────
    base_steps     = round(base_delta_c     * BASE_STEPS_PER_DEG)
    shoulder_steps = round(shoulder_delta_c * SHOULDER_STEPS_PER_DEG)
    elbow_steps    = round(elbow_delta_c    * ELBOW_STEPS_PER_DEG)
    wrist_steps    = round(wrist_delta_c    * WRIST_STEPS_PER_DEG)

    return IKResult(
        base_delta     = base_delta_c,
        shoulder_delta = shoulder_delta_c,
        elbow_delta    = elbow_delta_c,
        wrist_delta    = wrist_delta_c,
        base_steps     = base_steps,
        shoulder_steps = shoulder_steps,
        elbow_steps    = elbow_steps,
        wrist_steps    = wrist_steps,
        reachable      = reachable,
        warnings       = warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_tests():
    PASS = "✓ PASS"
    FAIL = "✗ FAIL"
    results = []

    def check(name, condition, detail=""):
        status = PASS if condition else FAIL
        results.append((status, name, detail))
        print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
        return condition

    print("\n" + "═"*60)
    print("IK SOLVER TEST SUITE")
    print("═"*60)

    # ──────────────────────────────────────────────────────────────
    # LEVEL 1: Known position verification
    # ──────────────────────────────────────────────────────────────
    print("\n── Level 1: Known Position Verification ──────────────────")

    # IK zero — FK gives us the exact tool tip position
    fk0 = forward(0, 0, 0)
    print(f"\n  IK zero FK result:\n{fk0}")

    r = solve(fk0.x, fk0.y, fk0.z)
    check("IK zero → all deltas ≈ 0°",
          abs(r.base_delta) < 0.5 and abs(r.shoulder_delta) < 0.5 and abs(r.elbow_delta) < 0.5,
          f"B={r.base_delta:.2f} S={r.shoulder_delta:.2f} E={r.elbow_delta:.2f}")
    check("IK zero → all steps = 0",
          r.base_steps == 0 and r.shoulder_steps == 0 and r.elbow_steps == 0,
          f"B={r.base_steps} S={r.shoulder_steps} E={r.elbow_steps}")
    check("IK zero → reachable", r.reachable)
    check("IK zero → no warnings", len(r.warnings) == 0,
          f"warnings: {r.warnings}" if r.warnings else "")

    # Max forward (shoulder ~90°, elbow ~-110°)
    fk_fwd = forward(0, 90, -110)
    print(f"\n  MAX FWD FK result:\n{fk_fwd}")
    r = solve(fk_fwd.x, fk_fwd.y, fk_fwd.z)
    check("MAX FWD → shoulder ≈ 90°",  abs(r.shoulder_delta - 90)  < 1.0, f"{r.shoulder_delta:.2f}°")
    check("MAX FWD → elbow ≈ -110°",   abs(r.elbow_delta - (-110)) < 1.0, f"{r.elbow_delta:.2f}°")
    check("MAX FWD → reachable", r.reachable)

    # Max high (shoulder ~0°, elbow ~-110°)
    fk_high = forward(0, 0, -110)
    r = solve(fk_high.x, fk_high.y, fk_high.z)
    check("MAX HIGH → shoulder ≈ 0°",  abs(r.shoulder_delta)        < 1.0, f"{r.shoulder_delta:.2f}°")
    check("MAX HIGH → elbow ≈ -110°",  abs(r.elbow_delta - (-110))  < 1.0, f"{r.elbow_delta:.2f}°")
    check("MAX HIGH → reachable", r.reachable)

    # Base rotation — X=200, Y=200
    r = solve(200, 200, 200)
    check("Base rotate → base ≈ -45°",
          abs(r.base_delta - (-45)) < 1.0, f"{r.base_delta:.2f}°")

    # Wrist auto-compensation — always shoulder_delta + elbow_delta
    r = solve(fk_fwd.x, fk_fwd.y, fk_fwd.z)
    expected_wrist = r.shoulder_delta + r.elbow_delta + WRIST_MOUNT_OFFSET
    check("Wrist = shoulder_delta + elbow_delta",
          abs(r.wrist_delta - expected_wrist) < 0.1,
          f"wrist={r.wrist_delta:.2f} expected={expected_wrist:.2f}")

    # ──────────────────────────────────────────────────────────────
    # LEVEL 2: FK/IK round-trip cross-check
    # ──────────────────────────────────────────────────────────────
    print("\n── Level 2: FK/IK Round-Trip Cross-Check ─────────────────")

    test_joints = [
        (0,   0,    0,   "IK zero"),
        (0,  90, -110,   "max forward"),
        (0,   0, -110,   "max high"),
        (0,  50,  -60,   "mid position"),
        (45, 30,  -80,   "base rotated"),
        (-30, 60, -90,   "base left"),
        (0,  95,  -80,   "near table"),
    ]

    max_error = 0.0
    all_roundtrip_pass = True
    for base_d, sh_d, el_d, label in test_joints:
        fk = forward(base_d, sh_d, el_d)
        ik = solve(fk.x, fk.y, fk.z)
        fk2 = forward(ik.base_delta, ik.shoulder_delta, ik.elbow_delta)
        error = math.sqrt((fk2.x-fk.x)**2 + (fk2.y-fk.y)**2 + (fk2.z-fk.z)**2)
        max_error = max(max_error, error)
        passed = error < 0.5
        if not passed:
            all_roundtrip_pass = False
        print(f"  {'✓' if passed else '✗'}  {label:20s}  FK→IK→FK error: {error:.3f}mm")

    check("All round-trips error < 0.5mm", all_roundtrip_pass, f"max error = {max_error:.3f}mm")

    # ──────────────────────────────────────────────────────────────
    # LEVEL 3: Workspace sweep — collision and limit detection
    # ──────────────────────────────────────────────────────────────
    print("\n── Level 3: Workspace & Collision Checks ─────────────────")

    # Too far
    r = solve(0, 450, 300)
    check("Too far → !reachable", not r.reachable)
    check("Too far → REACH warning", any("REACH" in w for w in r.warnings))

    # Wrist too low (below 17.5mm floor)
    r = solve(0, 200, 10)
    check("Z=10 → TABLE collision", any("TABLE" in w for w in r.warnings),
          f"warnings: {r.warnings}")

    # Just above floor — sh=82, el=-65 gives Z=40mm, well above 17.5mm floor
    fk_low = forward(0, 82, -65)
    r = solve(fk_low.x, fk_low.y, fk_low.z)
    check("Near-table valid position → reachable", r.reachable,
          f"Z={fk_low.z:.1f}mm  warnings: {r.warnings}")

    # Mount cylinder collision
    r = solve(0, 30, 50)
    check("Inside mount cylinder → ZONE1 or ZONE2 collision",
          any("ZONE" in w for w in r.warnings),
          f"warnings: {r.warnings}")

    # Singularity (directly above shoulder pivot)
    sh_x = SHOULDER_DH_OFFSET_MM   # shoulder is at (0, 44) at base=0
    r = solve(0, sh_x, 300)
    check("Above shoulder pivot → SINGULARITY warning",
          any("SINGULARITY" in w for w in r.warnings) or not r.reachable,
          f"warnings: {r.warnings}")

    # Soft limit — base too far: X=100, Y=-50 → base=-116.6° (outside ±110°)
    r = solve(100, -50, 200)
    check("Base > 110° → LIMIT warning", any("LIMIT [BASE]" in w for w in r.warnings),
          f"base={r.base_delta:.1f}°")

    # Soft limit — shoulder over max
    r = solve(0, 200, 5)
    check("Low Z → shoulder near limit", not r.reachable or r.shoulder_delta >= SHOULDER_MIN_DEG)

    # ──────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    passed = sum(1 for s, _, _ in results if s == PASS)
    total  = len(results)
    print(f"RESULT: {passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED — IK math is verified ✓")
    else:
        print("FAILURES DETECTED — review above ✗")
        for status, name, detail in results:
            if status == FAIL:
                print(f"  FAILED: {name}  [{detail}]")
    print("═"*60 + "\n")

    return passed == total


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nIK SOLVER — ARM_IK_CONVENTIONS_v4")
    print(f"  L1={L1}mm  L2={L2}mm  TOOL={TOOL_MM}mm")
    print(f"  Shoulder Z={SHOULDER_Z_OFFSET_MM}mm  DH={SHOULDER_DH_OFFSET_MM}mm")
    print(f"  BETA={math.degrees(BETA):.2f}°")
    print(f"  Sweep radius={SHOULDER_SWEEP_RADIUS_MM}mm")
    print(f"  Wrist floor={WRIST_HALF_HEIGHT_MM}mm")
    print(f"  Steps/deg: B={BASE_STEPS_PER_DEG:.3f}  S={SHOULDER_STEPS_PER_DEG:.3f}  "
          f"E={ELBOW_STEPS_PER_DEG:.3f}  W={WRIST_STEPS_PER_DEG:.3f}")

    _run_tests()
