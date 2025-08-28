import numpy as np
# -------------------------
# CONSTANTS
# -------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density at sea level (kg/m^3)

# -------------------------
# ROCKET + PAYLOAD PARAMETERS
# -------------------------
m_payload = 0.88        # payload mass (kg)
m_rocket = 3.0          # dry rocket + payload (example mass in kg, adjust!)
Cd_rocket = 0.75        # drag coefficient (cylinder nose approx)
A_rocket = np.pi * (0.138/2)**2   # cross-sectional area (m^2)

thrust = 200.0          # N (example motor thrust)
t_burn = 2.5            # burn time (s)
dt = 0.01               # timestep

# -------------------------
# ASCENT SIMULATION
# -------------------------
v = 0.0
h = 0.0
t = 0.0
mass = m_rocket

altitudes = []
velocities = []

burning = True
while True:
    # Thrust only during burn
    if t < t_burn:
        F_thrust = thrust
    else:
        F_thrust = 0
        burning = False

    # Drag
    F_drag = 0.5 * rho * Cd_rocket * A_rocket * v**2 * np.sign(v)

    # Net Force
    F_net = F_thrust - (mass * g) - F_drag
    a = F_net / mass

    # Update velocity and height
    v += a * dt
    h += v * dt
    t += dt

    altitudes.append(h)
    velocities.append(v)

    # Stop when it starts descending
    if not burning and v <= 0:
        break

apogee = h
print(f"Apogee reached at {apogee:.2f} m after {t:.2f} seconds.")

# -------------------------
# DESCENT (PARACHUTE)
# -------------------------
Cd_parachute = 1.5        # drag coefficient for round parachute
d_parachute = 0.5         # parachute diameter in m (adjust)
A_parachute = np.pi * (d_parachute/2)**2

# Terminal velocity
Vt = np.sqrt((2 * m_payload * g) / (rho * Cd_parachute * A_parachute))
t_descent = apogee / Vt

print(f"Descent with parachute: terminal velocity {Vt:.2f} m/s, descent time {t_descent:.2f} s.")

# -------------------------
# DRIFT CALCULATION (Wind)
# -------------------------
wind_speed = 5.0   # m/s horizontal wind
drift_distance = wind_speed * t_descent

print(f"Drift distance due to wind: {drift_distance:.2f} m.")
print(f"Predicted landing zone offset: {drift_distance:.2f} m downwind.")


