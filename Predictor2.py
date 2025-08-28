#!/usr/bin/env python3
"""
predict_from_telemetry.py

Reads cansat_telemetry(Sheet1) and:
 - for every telemetry row: compute a landing prediction (lat, lon)
 - final prediction using last row

Assumptions & notes:
 - Uses 'altitude' column for altitude (meters).
 - Uses successive lat/lon rows to estimate horizontal speed & bearing (proxy for wind).
 - Parachute is assumed to deploy at apogee. If currently ascending, apogee is estimated with vz^2/(2*g).
 - Terminal velocity computed using v_t = sqrt(2*m*g / (rho * Cd * A)).
 - Uses mean-air-density at mean altitude for descent (simple).
 - Default payload mass = 0.88 kg (880 g). Default parachute params are set but can be adjusted.
"""

import math
import pandas as pd
from datetime import datetime
import numpy as np

# --------------------
# Configurable params
# --------------------
EXCEL_PATH = r"C:\Users\chand\Downloads\cansat_telemetry.xlsx"
SHEET_NAME = "Sheet1"

# payload / parachute defaults (change if you have real values)
PAYLOAD_MASS = 0.88        # kg (880 g)
PARACHUTE_CD = 1.5         # parachute drag coefficient (typical round)
PARACHUTE_DIAMETER = 0.7   # meters (example). Change to your parachute diameter.
PARACHUTE_AREA = math.pi * (PARACHUTE_DIAMETER / 2.0) ** 2

# physics constants
G = 9.81                   # m/s^2
R_EARTH = 6371000.0        # m
RHO0 = 1.225               # kg/m^3 at sea level
SCALE_HEIGHT = 8500.0      # m for exponential atmosphere

# time parsing (timestamps in file are ISO-like)
TIME_COL = "timestamp"
LAT_COL = "lat"
LON_COL = "long"
ALT_COL = "altitude"       # explicit per user request

# output file
OUTPUT_CSV = "predictions_per_row.csv"


# --------------------
# Utility functions
# --------------------
def parse_time(s):
    # attempt to parse timestamp present in Excel (Z suffix)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

def haversine_distance_and_bearing(lat1, lon1, lat2, lon2):
    """
    Returns (distance_meters, bearing_deg_from_north_clockwise)
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    a = max(0.0, min(1.0, a))
    dist = 2 * R_EARTH * math.asin(math.sqrt(a))

    # bearing (forward azimuth)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0  # 0 = north, clockwise
    return dist, bearing

def destination_point(lat1, lon1, bearing_deg, distance_m):
    """
    Given start lat/lon (deg), initial bearing (deg) and distance (m),
    returns (lat2, lon2) in degrees using great-circle formulas.
    """
    ang = distance_m / R_EARTH
    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    theta = math.radians(bearing_deg)

    phi2 = math.asin(math.sin(phi1)*math.cos(ang) + math.cos(phi1)*math.sin(ang)*math.cos(theta))
    lambda2 = lambda1 + math.atan2(math.sin(theta)*math.sin(ang)*math.cos(phi1),
                                   math.cos(ang) - math.sin(phi1)*math.sin(phi2))
    return math.degrees(phi2), math.degrees(lambda2)

def air_density_at_alt(z):
    """
    Simple exponential model: rho = rho0 * exp(-z / H)
    """
    z_clamped = max(0.0, z)
    return RHO0 * math.exp(-z_clamped / SCALE_HEIGHT)

def terminal_velocity(mass, Cd, area, rho):
    """
    v_t = sqrt( 2*m*g / (rho * Cd * A) )
    """
    # prevent division by zero
    denom = max(1e-12, rho * Cd * area)
    return math.sqrt((2.0 * mass * G) / denom)


# --------------------
# Core predictor functions
# --------------------
def estimate_apogee(current_alt, vz):
    """
    If currently ascending (vz > 0), estimate apogee using simple kinematics:
      apogee = current_alt + vz^2 / (2*g)
    (neglects drag and thrust; simple approximation)
    """
    if vz <= 0:
        return current_alt
    return current_alt + (vz * vz) / (2.0 * G)

def predict_from_state(lat, lon, altitude, horiz_speed, horiz_bearing_deg,
                       payload_mass=PAYLOAD_MASS,
                       parachute_cd=PARACHUTE_CD,
                       parachute_area=PARACHUTE_AREA,
                       assume_parachute_opens_at_apogee=True,
                       current_vz=0.0):
    """
    Given a snapshot state (lat, lon, altitude, horizontal speed+bearing),
    compute predicted landing point assuming parachute opens at apogee.
    horiz_speed: m/s (estimated from gps movement)
    horiz_bearing_deg: direction of travel / wind (0 = north)
    Returns a dict with vt, desc_time, drift_m, pred_lat, pred_lon
    """
    # 1) estimate apogee altitude if ascending
    apogee_alt = estimate_apogee(altitude, current_vz) if assume_parachute_opens_at_apogee else altitude

    # 2) compute representative altitude for air density (use mean of apogee->ground)
    mean_alt = max(0.0, (apogee_alt / 2.0))
    rho_mean = air_density_at_alt(mean_alt)

    # 3) terminal velocity under parachute (using mean rho)
    vt = terminal_velocity(payload_mass, parachute_cd, parachute_area, rho_mean)

    # 4) descent time (simple: distance / vt)
    descent_time = apogee_alt / vt if vt > 0 else float('inf')

    # 5) horizontal drift: use measured horizontal speed as proxy for wind (if you have wind, pass it here)
    drift = horiz_speed * descent_time

    # 6) predicted lat/lon (move along bearing by "drift" meters)
    pred_lat, pred_lon = destination_point(lat, lon, horiz_bearing_deg, drift)

    return {
        "apogee_alt": apogee_alt,
        "rho_mean": rho_mean,
        "vt": vt,
        "descent_time_s": descent_time,
        "drift_m": drift,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon
    }


# --------------------
# Main: load telemetry, iterate rows
# --------------------
def main():
    # load telemetry
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    # require timestamp, lat, long, altitude
    if not {TIME_COL, LAT_COL, LON_COL, ALT_COL}.issubset(df.columns):
        raise SystemExit(f"Excel must contain columns: {TIME_COL}, {LAT_COL}, {LON_COL}, {ALT_COL}")

    # parse time to datetime (if possible)
    df[TIME_COL] = df[TIME_COL].astype(str).apply(parse_time)
    # sort by time just in case
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Prepare output
    rows_out = []
    prev_lat = None
    prev_lon = None
    prev_time = None

    for idx, r in df.iterrows():
        lat = float(r[LAT_COL])
        lon = float(r[LON_COL])
        alt = float(r[ALT_COL])

        # time delta
        t = r[TIME_COL]
        if prev_time is None or t is None:
            dt = None
        else:
            dt = (t - prev_time).total_seconds() if hasattr(t, "timestamp") else None

        # estimate horizontal speed & bearing from previous GPS (ground speed proxy)
        if prev_lat is None or prev_lon is None or dt is None or dt <= 0:
            horiz_speed = 0.0
            horiz_bearing = 0.0
        else:
            dist_m, bearing = haversine_distance_and_bearing(prev_lat, prev_lon, lat, lon)
            horiz_speed = dist_m / dt
            horiz_bearing = bearing

        # estimate vertical speed vz (m/s) from altitude delta (positive = ascending)
        if prev_time is None or dt is None or dt <= 0:
            vz = 0.0
        else:
            vz = (alt - prev_alt) / dt if 'prev_alt' in locals() else 0.0

        # predict landing from this snapshot
        pred = predict_from_state(
            lat, lon, alt,
            horiz_speed, horiz_bearing,
            payload_mass=PAYLOAD_MASS,
            parachute_cd=PARACHUTE_CD,
            parachute_area=PARACHUTE_AREA,
            assume_parachute_opens_at_apogee=True,
            current_vz=vz
        )

        # collect outputs
        rows_out.append({
            "row_idx": idx,
            "timestamp": t,
            "lat": lat, "lon": lon, "alt": alt,
            "horiz_speed_mps": horiz_speed,
            "horiz_bearing_deg": horiz_bearing,
            "vz_mps": vz,
            "apogee_alt_m": pred["apogee_alt"],
            "rho_mean": pred["rho_mean"],
            "vt_mps": pred["vt"],
            "descent_time_s": pred["descent_time_s"],
            "drift_m": pred["drift_m"],
            "pred_lat": pred["pred_lat"],
            "pred_lon": pred["pred_lon"]
        })

        # update previous
        prev_lat = lat
        prev_lon = lon
        prev_time = t
        prev_alt = alt

    # write CSV for per-row predictions
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Per-row predictions saved to: {OUTPUT_CSV}")

    # final landing prediction using last row
    last = out_df.iloc[-1]
    print("\n=== FINAL PREDICTION (using last telemetry row) ===")
    print(f"time: {last['timestamp']}")
    print(f"current position: {last['lat']:.6f}, {last['lon']:.6f}, alt={last['alt']:.2f} m")
    print(f"estimated apogee: {last['apogee_alt_m']:.2f} m")
    print(f"parachute terminal velocity (m/s): {last['vt_mps']:.2f}")
    print(f"descent time (s): {last['descent_time_s']:.1f}")
    print(f"predicted drift (m): {last['drift_m']:.1f}")
    print(f"predicted landing: {last['pred_lat']:.6f}, {last['pred_lon']:.6f}")

if __name__ == "__main__":
    main()
